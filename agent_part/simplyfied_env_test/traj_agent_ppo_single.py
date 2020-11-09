import logging
import argparse
import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent, Agent, StayAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from utils import traj2demo, demo2traj

import overcooked_gym_env


class NNAgent(Agent):
    def __init__(self, mdp, model, horizon):
        super().__init__()
        self.mdp = mdp
        self.model = model
        self.horizon = horizon

    def action(self, state):
        obs = self.mdp.lossless_state_encoding(state, horizon=self.horizon)
        # action, value = self.model.action_value(obs[0][None, :].astype(np.float32))
        probs, _ = self.model.predict_on_batch(obs[0][None, :].astype(np.float32))
        action = np.random.choice(6, p=probs[0])
        action = Action.INDEX_TO_ACTION[action]
        return action, {}


class PPO:
    def __init__(self, state_dim, action_dim, lr, clip_ratio, entropy_c):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.entropy_c = entropy_c
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.RMSprop(lr)

    def create_model(self):
        inputs = Input(self.state_dim)
        x = Conv2D(64, 3, padding='same', activation=tf.nn.leaky_relu)(inputs)
        x = Conv2D(64, 3, padding='same', activation=tf.nn.leaky_relu)(x)
        x = Flatten()(x)

        x1 = Dense(128, activation='relu')(x)
        logits = Dense(self.action_dim, activation='softmax')(x1)

        x2 = Dense(64, activation='relu')(x)
        value = Dense(1)(x2)

        return tf.keras.Model(inputs=inputs, outputs=[logits, value])

    def compute_actor_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        entropy = tf.reduce_mean(new_policy * tf.math.log(new_policy))
        return tf.reduce_mean(surrogate) + self.entropy_c * entropy

    def compute_critic_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, old_policy, states, actions, gaes, td_targets):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float32)

        with tf.GradientTape() as tape:
            probs, v_pred = self.model(states, training=True)
            actor_loss = self.compute_actor_loss(old_policy, probs, actions, gaes)

            assert v_pred.shape == td_targets.shape
            critic_loss = self.compute_critic_loss(v_pred, tf.stop_gradient(td_targets))

            loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return actor_loss, critic_loss


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, lmbda=0.95, clip_ratio=0.2,
                 entropy_c=1e-4, reward_shaping_horizon=2000):
        self.state_dim = list(state_dim.shape)
        self.action_dim = action_dim.n

        self.gamma = gamma
        self.lmbda = lmbda
        self.reward_shaping_horizon = reward_shaping_horizon

        self.nb_episodes = 0

        self.ppo = PPO(self.state_dim, self.action_dim, lr, clip_ratio, entropy_c)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros_like(rewards, dtype=np.float32)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        # print(list)
        # batch = list[0]
        # for elem in list[1:]:
        #     batch = np.append(batch, elem, axis=0)
        # print(111)
        # print(batch.shape)
        # print(np.array(list).shape)
        # return batch.astype(np.float32)
        return np.squeeze(np.array(list), axis=1)

    def train(self, env, max_episodes=1000, ep_in_batch=3, epochs=5):

        episode_reward = [0.0]
        old_policys_all = []
        states_all = []
        actions_all = []
        gaes_all = []
        td_targets_all = []
        # time0 = time.time()
        for ep in range(max_episodes):
            # time1 = time.time()
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []
            v_values_batch = []

            episode_reward.append(0.0)
            done = False

            state = env.reset()

            while not done:

                probs, old_value = self.ppo.model.predict_on_batch(np.reshape(state, [1] + self.state_dim))
                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, env_info = env.step(action)

                if self.reward_shaping_horizon > 0:
                    shaping_reward = sum(env_info["shaped_r_by_agent"]) * max([0, 1-self.nb_episodes/self.reward_shaping_horizon])
                    reward += shaping_reward - 0.02

                state = np.reshape(state, [1] + self.state_dim)
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1] + self.state_dim)
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(probs)
                v_values_batch.append(old_value)

                if done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)
                    v_values = self.list_to_batch(v_values_batch)

                    _, next_v_value = self.ppo.model.predict_on_batch(next_state)

                    gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

                    old_policys_all.append(old_policys)
                    states_all.append(states)
                    actions_all.append(actions)
                    gaes_all.append(gaes)
                    td_targets_all.append(td_targets)

                    if len(states_all) == ep_in_batch:
                        old_policys = np.concatenate(old_policys_all, axis=0)
                        states = np.concatenate(states_all, axis=0)
                        actions = np.concatenate(actions_all, axis=0)
                        gaes = np.concatenate(gaes_all, axis=0)
                        td_targets = np.concatenate(td_targets_all, axis=0)

                        for epoch in range(epochs):
                            actor_loss, critic_loss = self.ppo.train(old_policys, states, actions, gaes, td_targets)
                            logging.debug("[%d/%d] Losses: %s %s" % (ep + 1, max_episodes, actor_loss, critic_loss))

                        old_policys_all = []
                        states_all = []
                        actions_all = []
                        gaes_all = []
                        td_targets_all = []

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []
                    v_values_batch = []

                episode_reward[-1] += reward[0][0]
                state = next_state[0]

            # print('EP{} EpisodeReward={}'.format(ep, episode_reward[-1]))
            logging.info("Episode: %03d, Reward: %.2f" % (ep, episode_reward[-1]))
            self.nb_episodes += 1

        return episode_reward

    def test(self, env, filename, nb_game=1, render=False):
        a0 = NNAgent(env.mdp, self.ppo.model, env.horizon)
        a1 = StayAgent()
        agent_pair = AgentPair(a0, a1)
        # trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

        ep_rewards = []

        for i in range(nb_game):
            trajectories = env.get_rollouts(agent_pair, 1)
            trajectories = AgentEvaluator.make_trajectories_json_serializable(trajectories)

            ep_rewards.append(sum(trajectories["ep_rewards"][0]))

            with open("trajs/" + filename + str(i) + ".json", "w") as f:
                json.dump(traj2demo(trajectories), f)

        return ep_rewards

    def save_model(self):
        if not os.path.exists("models"):
            os.mkdir("models")
        if not os.path.exists("models/ppo_s"):
            os.mkdir("models/ppo_s")
        self.ppo.model.save_weights("models/ppo/ppo_{}_weights".format(self.nb_episodes))

    def load_model(self, path):
        self.ppo.model.load_weights(path)


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.92)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--clip_ratio', type=float, default=0.2)
parser.add_argument('--lmbda', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--entropy_c', type=float, default=1e-2)
parser.add_argument('-b', '--ep_in_batch', type=int, default=4)
parser.add_argument('-n', '--num_episodes', type=int, default=1500)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

args = parser.parse_args()

if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 1,
        "DISH_PICKUP_REWARD": 0,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    train_env = overcooked_gym_env.get_gym_env(layout_name="cramped_room_o_3orders", horizon=1000,
                                               params_to_overwrite={"rew_shaping_params": rew_shaping_params})
    test_env = overcooked_gym_env.get_gym_env(layout_name="cramped_room_o_3orders", horizon=1000)
    agent = PPOAgent(train_env.observation_space, train_env.action_space, args.lr,
                     gamma=args.gamma, clip_ratio=args.clip_ratio, lmbda=args.lmbda, entropy_c=args.entropy_c)

    # agent.train_bc(train_env.base_env, "trajs/10_10_2020_19_42_3_ppo_bc_1_long.json", epochs=500,
    #                batch_size=args.batch_size)
    # print("Behavior cloning finished in {} min".format((time.time()-start_time)//60))

    with open("rewards/traj_ac_rewards.txt", "w") as reward_output:
        # rewards = agent.test(test_env.base_env, "traj_ac", 3)
        # reward_output.write(str(sorted(rewards)))

        rewards_history_all = []

        for i in range(4):
            rewards_history = agent.train(train_env, max_episodes=args.num_episodes, ep_in_batch=args.ep_in_batch,
                                          epochs=args.epochs)
            rewards_history_all += rewards_history
            print("Finished training. Testing...")
            rewards = agent.test(test_env.base_env, "traj_ppo", 3)
            reward_output.write(str(sorted(rewards)))
            print("Training round {} finished in {} min".format(i, (time.time() - start_time) // 60))
            agent.save_model()

    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history_all), 10), rewards_history_all[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
