import logging
import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

import overcooked_gym_env


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        print("num_actions", num_actions)
        # Note: no tf.get_variable(), just simple Keras API!
        self.conv1 = kl.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu)
        self.conv2 = kl.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu)
        self.flatten1 = kl.Flatten()
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(64, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        # x = tf.convert_to_tensor(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten1(x)
        # Separate hidden layers from the same input tensor.
        # x = self.hidden1(x)
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        # logits = self.predict_on_batch(obs)
        # print(logits)
        action = self.dist.predict_on_batch(logits)
        # print(action)
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def action_value_full(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # print(222)
        # print(action.shape)
        # print(value.shape)
        return action, np.squeeze(value, axis=-1)


class NNAgent(Agent):
    def __init__(self, mdp, model, horizon):
        super().__init__()
        self.mdp = mdp
        self.model = model
        self.horizon = horizon

    def action(self, state):
        obs = self.mdp.lossless_state_encoding(state, horizon=self.horizon)
        # action, value = self.model.action_value(obs[0][None, :].astype(np.float32))
        action, _ = self.model.action_value(obs[0][None, :].astype(np.float32))
        action = Action.INDEX_TO_ACTION[action]
        return action, {}


class A2CAgent:
    def __init__(self, model, lr=5e-5, gamma=0.99, value_c=0.5, entropy_c=1e-4, reward_shaping_horizon=2000):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.reward_shaping_horizon = reward_shaping_horizon
        self.nb_episodes = 0

        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            # Define separate losses for policy logits and value estimate.
            loss=[self._logits_loss, self._value_loss])
            # loss='sparse_categorical_crossentropy')

    def train_bc(self, env, filename, batch_size=1200, epochs=10):
        traj = demo2traj(filename)
        observations, rewards, dones, actions, next_observations = self.traj2data(traj, env)
        # print(observations.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # print(actions.shape)
        # print(next_observations.shape)

        _, values = self.model.action_value_full(observations)
        _, next_values = self.model.action_value_full(next_observations)

        returns, advs = self._returns_advantages_bc(rewards, dones, values, next_values)
        # print(actions.shape)
        # print(advs.shape)
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        # Performs a full training step on the collected batch.
        # Note: no need to mess around with gradients, Keras API handles it.
        self.model.fit(observations, [acts_and_advs, returns], epochs=epochs, batch_size=batch_size)

    def train_rl(self, env, batch_sz=1200, updates=250):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], env_info = env.step(actions[step])

                if self.reward_shaping_horizon > 0:
                    shaping_reward = sum(env_info["shaped_r_by_agent"]) * max([0, 1-self.nb_episodes/self.reward_shaping_horizon])
                    rewards[step] += shaping_reward

                ep_rewards[-1] += rewards[step]

                if dones[step]:
                    self.nb_episodes += 1
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %.2f" % (len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages_rl(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        return ep_rewards

    def test(self, env, filename, nb_game=1, render=False):
        a0 = NNAgent(env.mdp, self.model, env.horizon)
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

    def traj2data(self, trajectories, env):
        action2index = {
            (0, -1): 0,
            (0, 1): 1,
            (1, 0): 2,
            (-1, 0): 3,
            (0, 0): 4,
            "interact": 5
        }
        actions = []
        rewards = []
        dones = []
        observations = []
        next_observations = []

        for ep_list in trajectories["ep_states"]:
            ep_list[0]._all_orders = [
                {
                    "ingredients": [
                        "onion",
                        "onion",
                        "onion"
                    ]
                }
            ]
            ep_list[0].timestep = 0
            observation = env.mdp.lossless_state_encoding(ep_list[0], horizon=env.horizon)[0]
            observations.append(observation)
            for i in range(1, len(ep_list)):
                ep_list[i]._all_orders = [
                    {
                        "ingredients": [
                            "onion",
                            "onion",
                            "onion"
                        ]
                    }
                ]
                ep_list[i].timestep = i
                observation = env.mdp.lossless_state_encoding(ep_list[i], horizon=env.horizon)[0]
                if i != len(ep_list) - 1:
                    observations.append(observation)
                next_observations.append(observation)

        for ep_list in trajectories["ep_actions"]:
            for j_a in ep_list:
                actions.append(action2index[j_a[0]])
            actions.pop(-1)

        for ep_list in trajectories["ep_rewards"]:
            rewards += ep_list[:-1]
            dones += [False for _ in range(len(ep_list)-1)]
            dones[-1] = True

        # print(np.array(observations, dtype=np.float32).shape)
        return np.array(observations, dtype=np.float32), np.array(rewards), np.array(dones), np.array(actions), np.array(next_observations, dtype=np.float32)

    def _returns_advantages_bc(self, rewards, dones, values, next_values):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        # returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        returns = np.zeros_like(rewards)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            if dones[t]:
                returns[t] = rewards[t]
            elif t != rewards.shape[0] - 1 and dones[t+1]:
                returns[t] = rewards[t] + self.gamma * next_values[t]
            else:
                returns[t] = rewards[t] + self.gamma * returns[t + 1]
        # returns = returns[:-1]
        # Advantages are equal to returns - baseline (value estimates in our case).
        # print(1111)
        # print(returns.shape)
        # print(values.shape)
        advantages = returns - values
        return returns, advantages

    def _returns_advantages_rl(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    def _cross_entropy_loss(self, actions, logits):
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits)
        # Entropy loss can be calculated as cross-entropy over itself.
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss

    def save_model(self):
        if not os.path.exists("models"):
            os.mkdir("models")
        self.model.save("models/ac_{}.h5".format(self.nb_episodes))

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=3000)
    parser.add_argument('-n', '--num_updates', type=int, default=500)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-ga', '--gamma', type=float, default=0.92)
    parser.add_argument('-r', '--render_test', action='store_true', default=False)
    parser.add_argument('-p', '--plot_results', action='store_true', default=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

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
    model = Model(num_actions=train_env.action_space.n)
    agent = A2CAgent(model, args.learning_rate, gamma=args.gamma)

    # agent.train_bc(train_env.base_env, "trajs/10_10_2020_19_42_3_ppo_bc_1_long.json", epochs=500, batch_size=args.batch_size)

    with open("rewards/traj_ac_rewards.txt", "w") as reward_output:
        rewards = agent.test(test_env.base_env, "traj_ac", 3)
        reward_output.write(str(sorted(rewards)))

        rewards_history_all = []

        for i in range(5):
            rewards_history = agent.train_rl(train_env, args.batch_size, args.num_updates)
            rewards_history_all += rewards_history
            print("Finished training. Testing...")
            rewards = agent.test(test_env.base_env, "traj_ac", 3)
            reward_output.write(str(sorted(rewards)))
            agent.save_model()

    if args.plot_results:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history_all), 10), rewards_history_all[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
