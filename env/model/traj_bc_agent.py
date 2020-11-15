import logging
import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import itertools
import json
from collections import defaultdict, Counter


class TrajBCAgent:
    def __init__(self, lr=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


    def train(self, batch_sz=1200, epochs=10):
        traj = demo2traj(filename)
        observations, rewards, dones, actions = self.traj2data(traj, env)
        print(observations.shape)
        self.model.fit(observations, actions, epochs=epochs)

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

        for ep_list in trajectories["ep_states"]:
            for i in range(len(ep_list)):
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
                observations.append(observation)

        for ep_list in trajectories["ep_actions"]:
            for j_a in ep_list:
                actions.append(action2index[j_a[0]])

        for ep_list in trajectories["ep_rewards"]:
            rewards += ep_list
            dones.append([False for _ in range(len(ep_list))])
            dones[-1] = True

        return np.array(observations, dtype=np.float32), np.array(rewards), dones, np.array(actions)

    def action(self, state):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1200)
    parser.add_argument('-n', '--num_updates', type=int, default=2500)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-6)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room_o_3orders", horizon=1000)
    agent = TrajBCAgent(args.learning_rate)

    agent.train(env.base_env, "trajs/traj_ac0.json", epochs=500)
    # rewards_history = agent.train(env, args.batch_size, args.num_updates)
    # print("Finished training. Testing...")
    # print("Total Episode Reward: %d" % agent.test(env, False))
    #
    # if args.plot_results:
    #     plt.style.use('seaborn')
    #     plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
    #     plt.xlabel('Episode')
    #     plt.ylabel('Total Reward')
    #     plt.show()
