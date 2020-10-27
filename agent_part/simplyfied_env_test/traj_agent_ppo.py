import gym
import argparse
import numpy as np

import logging
import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS, Overcooked
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent, Agent, StayAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir, load_from_json, save_as_json

from utils import traj2demo, demo2traj

import itertools
import json
from collections import defaultdict, Counter
import gym
import time

import overcooked_gym_env

# tf.keras.backend.set_floatx('float32')


class NNAgent(Agent):
    def __init__(self, mdp, model, horizon):
        super().__init__()
        self.mdp = mdp
        self.model = model
        self.horizon = horizon

    def action(self, state):
        obs = self.mdp.lossless_state_encoding(state, horizon=self.horizon)
        # action, value = self.model.action_value(obs[0][None, :].astype(np.float32))
        probs = self.model.predict(obs[0][None, :].astype(np.float32))
        action = np.random.choice(6, p=probs[0])
        action = Action.INDEX_TO_ACTION[action]
        return action, {}


class Actor:
    # TODO entropy reg
    def __init__(self, state_dim, action_dim, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input(self.state_dim),
            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(64, 3, padding='same', activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, old_policy, new_policy, actions, gaes):
        gaes = tf.stop_gradient(gaes)
        old_log_p = tf.math.log(tf.reduce_sum(old_policy * actions))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(new_policy * actions))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, states, actions, gaes):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float32)

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(old_policy, logits, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    # TODO clip
    def __init__(self, state_dim, lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input(self.state_dim),
            Conv2D(64, 3, padding='same', activation='relu'),
            Conv2D(64, 3, padding='same', activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_a=1e-4, lr_c=5e-5, gamma=0.99, lmbda=0.95, entropy_c=1e-4, reward_shaping_horizon=2000):
        self.state_dim = list(state_dim.shape)
        self.action_dim = action_dim.n

        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_c = entropy_c
        self.reward_shaping_horizon = reward_shaping_horizon

        self.nb_episodes = 0

        self.actor = Actor(self.state_dim, self.action_dim, lr_a)
        self.critic = Critic(self.state_dim, lr_c)

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
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch.astype(np.float32)

    def train(self, env, max_episodes=1000, batch_sz=1200, epochs=5):
        # TODO multi epi batch
        episode_reward = [0.0]
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward.append(0.0)
            done = False

            state = env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.model.predict(
                    np.reshape(state, [1] + self.state_dim))
                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, env_info = env.step(action)

                if self.reward_shaping_horizon > 0:
                    shaping_reward = sum(env_info["shaped_r_by_agent"]) * max([0, 1-self.nb_episodes/self.reward_shaping_horizon])
                    reward += shaping_reward

                state = np.reshape(state, [1] + self.state_dim)
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1] + self.state_dim)
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(probs)

                if len(state_batch) >= batch_sz or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)
                        logging.debug("[%d/%d] Losses: %s %s" % (ep + 1, max_episodes, actor_loss, critic_loss))

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward[-1] += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(ep, episode_reward[-1]))
            logging.info("Episode: %03d, Reward: %.2f" % (ep, episode_reward[-1]))
            self.nb_episodes += 1
        return episode_reward

    def test(self, env, filename, nb_game=1, render=False):
        a0 = NNAgent(env.mdp, self.actor.model, env.horizon)
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
        self.actor.model.save_weights("models/ppo_actor_{}_weights".format(self.nb_episodes))
        self.critic.model.save_weights("models/ppo_critic_{}_weights".format(self.nb_episodes))

    def load_model(self, actor_path, critic_path):
        self.actor.model.load_weights(actor_path)
        self.critic.model.load_weights(critic_path)


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.92)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=5e-4)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--clip_ratio', type=float, default=0.2)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('-b', '--batch_size', type=int, default=3000)
parser.add_argument('-n', '--num_episodes', type=int, default=1500)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

args = parser.parse_args()

if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logging.getLogger().setLevel(logging.INFO)

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
    agent = PPOAgent(train_env.observation_space, train_env.action_space, args.actor_lr, args.critic_lr, gamma=args.gamma)

    # agent.train_bc(train_env.base_env, "trajs/10_10_2020_19_42_3_ppo_bc_1_long.json", epochs=500, batch_size=args.batch_size)
    # print("Behavior cloning finished in {} min".format((time.time()-start_time)//60))

    with open("rewards/traj_ac_rewards.txt", "w") as reward_output:
        # rewards = agent.test(test_env.base_env, "traj_ac", 3)
        # reward_output.write(str(sorted(rewards)))

        rewards_history_all = []

        for i in range(4):
            rewards_history = agent.train(train_env, args.num_episodes, args.batch_size)
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
