import argparse
import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as kl
import overcooked_gym_env
from overcooked_ai_py.mdp.actions import Action
from il_model_utils import load_from_json, NullContextManager, TfContextManager, TRAINING_PARAMS, build_model, save_model, load_model
from traj_agent_ac import A2CAgent, Model

logging.getLogger().setLevel(logging.DEBUG)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    return parser.parse_args()


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = kl.Dense( units=20, activation=tf.nn.leaky_relu, name='layer1')
        self.layer_2 = kl.Dense(units=20, activation=tf.nn.leaky_relu, name='layer2')
        self.layer_3 = kl.Dense(units=20, activation=tf.nn.leaky_relu, name='layer3')
        self.layer_prob = kl.Dense(units=1, activation=tf.nn.sigmoid, name='prob')
                                  
    def get_prob(self, s_a):
        x = self.layer_1(s_a)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_prob(x)
        
    def get_rewards(self, s_a):
        prob = self.get_prob(s_a)
        return 20 + tf.math.log(prob) #clip?
    
    def get_loss(self, inputs):
        prob_expert = self.get_prob(inputs[0])
        prob_agent = self.get_prob(inputs[1])
        loss_expert = tf.reduce_mean(tf.math.log(prob_expert))
        loss_agent = tf.reduce_mean(tf.math.log(1 - prob_agent))
        loss = -(loss_agent + loss_expert)
        return loss
    
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.get_loss(inputs[0])
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


def concat_s_a(s, a):
    s = tf.reshape(s, shape=(s.shape[0], -1))
    a = tf.one_hot(a, Action.NUM_ACTIONS)
    a = tf.reshape(a, shape=(a.shape[0], -1))
    return tf.concat([s,a], axis=-1)


def train(args):  
    
    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
    # model
    model = Model(num_actions=len(Action.ALL_ACTIONS))
    def reassign_weights(model):
        # BC Model
        bc_model = load_model('./bc_run')
        """
        # for var in bc_model.trainable_variables:
        #     print(var.name)
        ac_model/conv2d/kernel:0
        ac_model/conv2d/bias:0
        ac_model/conv2d_1/kernel:0
        ac_model/conv2d_1/bias:0
        ac_model/dense/kernel:0
        ac_model/dense/bias:0
        ac_model/dense_1/kernel:0
        ac_model/dense_1/bias:0
        ac_model/policy_logits/kernel:0
        ac_model/policy_logits/bias:0
        """
        
        conv1_kernel_init = tf.constant_initializer(bc_model.trainable_variables[0].value().numpy())
        conv1_bias_init = tf.constant_initializer(bc_model.trainable_variables[1].value().numpy())
        model.conv1 = kl.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer=conv1_kernel_init, bias_initializer=conv1_bias_init)
        
        conv2_kernel_init = tf.constant_initializer(bc_model.trainable_variables[2].value().numpy())
        conv2_bias_init = tf.constant_initializer(bc_model.trainable_variables[3].value().numpy())
        model.conv2 = kl.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer=conv2_kernel_init, bias_initializer=conv2_bias_init)
        
        dense1_kernel_init = tf.constant_initializer(bc_model.trainable_variables[4].value().numpy())
        dense1_bias_init = tf.constant_initializer(bc_model.trainable_variables[5].value().numpy())
        model.hidden1 = kl.Dense(128, activation='relu',
                                kernel_initializer=dense1_kernel_init, bias_initializer=dense1_bias_init)
        
        dense2_kernel_init = tf.constant_initializer(bc_model.trainable_variables[6].value().numpy())
        dense2_bias_init = tf.constant_initializer(bc_model.trainable_variables[7].value().numpy())
        model.hidden2 = kl.Dense(64, activation='relu',
                                kernel_initializer=dense2_kernel_init, bias_initializer=dense2_bias_init)
        
        policy_kernel_init = tf.constant_initializer(bc_model.trainable_variables[8].value().numpy())
        policy_bias_init = tf.constant_initializer(bc_model.trainable_variables[9].value().numpy())
        model.logits = kl.Dense(len(Action.ALL_ACTIONS), name='policy_logits', #activation='sigmoid',
                                kernel_initializer=policy_kernel_init, bias_initializer=policy_bias_init)
        
        return model
    
    model = reassign_weights(model)
    # Agent
    agent = A2CAgent(model=model, gamma=args.gamma)
    
    # Discriminator
    D = Discriminator()
    D.compile(K.optimizers.Adam(TRAINING_PARAMS["learning_rate"]))
    #
    obs = env.reset()
    success_num = 0
    #load expert observations and actions
    _, expert_observations, expert_actions = load_from_json()
    expert_observations = expert_observations[:400,]
    expert_actions = expert_actions[:400,]
    ep_rewards = [0.0]
    
    
    writer = tf.summary.create_file_writer("./gail_logs")
    with writer.as_default():
        for iterations in range(args.iteration):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            dones = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1
                act, v_pred = agent.model.action_value(obs[None, :])
                act, v_pred = act.item(), v_pred.item()

                next_obs,reward,done,info = env.step(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)
                dones.append(done)

                if done:
                    agent.nb_episodes += 1
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %.2f" % (len(ep_rewards) - 1, ep_rewards[-2]))
                    break
                else:
                    obs = next_obs


            if sum(rewards) >= 20:
                success_num += 1
                if success_num >= 25:
                    agent.save_model()
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            observations = np.asarray(observations)
            actions = np.array(actions).astype(dtype = np.int32)

            D.fit([concat_s_a(expert_observations, expert_actions),
                    concat_s_a(observations, actions)], epochs=2)
            d_loss = D.get_loss([concat_s_a(expert_observations, expert_actions),
                    concat_s_a(observations, actions)])
            
            d_rewards = D.get_rewards(concat_s_a(observations, actions))
            _, next_value = agent.model.action_value(next_obs[None, :])
            returns, advs = agent._returns_advantages_rl(np.asarray(rewards), np.asarray(dones), v_preds, next_value)
            # acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            acts_and_Dvals = np.concatenate([actions[:, None], d_rewards], axis=-1)
            # train policy
            for epoch in range(5):
                # a_losses = agent.model.train_on_batch(observations, [acts_and_advs, returns])
                a_losses = agent.model.train_on_batch(observations, [acts_and_Dvals, returns])
            
            logging.debug("[%d/%d] Losses: %s" % (iterations + 1, args.iteration, a_losses))

            tf.summary.scalar("Discriminator loss", d_loss, step=iterations)
            tf.summary.scalar("Agent logit loss", a_losses[1], step=iterations)
            tf.summary.scalar("Agent value loss", a_losses[2], step=iterations)

if __name__ == '__main__':
    args = argparser()
    train(args)