import argparse
import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as kl
import overcooked_gym_env
from overcooked_ai_py.mdp.actions import Action
from il_model_utils import load_from_json, NullContextManager, TfContextManager, TRAINING_PARAMS, build_model, save_model


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e1))
    return parser.parse_args()

class Actor2Critic:
    def __init__(self):
        self.action_model = self.construct_network()
        self.critic_model = self.construct_network()
        self.action_model.compile(optimizer=K.optimizers.Adam(TRAINING_PARAMS["learning_rate"]))
        self.critic_model.compile(optimizer=K.optimizers.Adam(TRAINING_PARAMS["learning_rate"]))
        
    def construct_network(self):
        return tf.keras.Sequential([
            kl.Input(self.state_dim),
            kl.Conv2D(64, 3, padding='same', activation='relu'),
            kl.Conv2D(64, 3, padding='same', activation='relu'),
            kl.Flatten(),
            kl.Dense(64, activation='relu'),
            kl.Dense(self.action_dim, activation='softmax')
        ])
        

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
        return tf.math.log(prob) #clip?
    
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            prob_expert = self.get_prob(inputs[0][0])
            prob_agent = self.get_prob(inputs[0][1])
            loss_expert = tf.reduce_mean(tf.math.log(prob_expert))
            loss_agent = tf.reduce_mean(tf.math.log(1 - prob_agent))
            loss = -(loss_agent + loss_expert)
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
    # Agent
    A2C = Actor2Critic()
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

    for iterations in range(args.iteration):
        observations = []
        actions = []
        rewards = []
        v_preds = []
        dones = []
        ep_rewards = [0.0]
        run_policy_steps = 0
        while True:
            run_policy_steps += 1
            act = A2C.get_action(obs[None, :])
            v_pred = A2C.get_value(concat_s_a(obs, act))

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


        if sum(rewards) >= 195:
            success_num += 1
            if success_num >= 25:
                agent.save_model()
                print('Clear!! Model saved.')
                break
        else:
            success_num = 0

        observations = np.asarray(observations)
        actions = np.array(actions).astype(dtype = np.int32)

        D.fit([concat_s_a(expert_observations, expert_actions), concat_s_a(observations, actions)], epochs=2)

        d_rewards = D.get_rewards(concat_s_a(observations, actions))

        _, next_value = agent.model.action_value(next_obs[None, :])
        returns, advs = agent._returns_advantages_rl(rewards, dones, v_preds, next_value)
        # A trick to input actions and advantages through same API.
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        

        # train policy
        for epoch in range(5):
            losses = agent.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (iterations + 1, args.iteration, losses))


if __name__ == '__main__':
    args = argparser()
    train(args)