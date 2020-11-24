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


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class TrajBCAgent:
    def __init__(self, env, lr=1e-4, test=False):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.model_dir = "traj_bc_model"
        self.env = env

        if test:
            self.model = tf.keras.models.load_model(self.model_dir)

        else:
            self.model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(6, activation="softmax"),
                ]
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

    def train(self, filename, encoding_fn, batch_sz=32, epochs=10):
        observations, rewards, dones, actions = self.traj2data(filename, encoding_fn)
        print(observations.shape)
        self.model.fit(observations, actions, batch_size=batch_sz, epochs=epochs)
        self.model.save(self.model_dir)

    def traj2data(self, traj_file, encoding_fn):
        with open(traj_file) as f:
            trajectories = list(map(json.loads, f.readlines()))

        action2index = {"U": 0, "R": 1, "D": 2, "L": 3, "I": 4, "C": 5}
        actions = []
        rewards = []
        dones = []
        observations = []

        for step_dict in trajectories:
            observation = encoding_fn(self.env, step_dict)
            observations.append(observation)

            rewards.append(step_dict["reward"])
            dones.append(False)
            actions.append(action2index[step_dict["action"]])

        dones[-1] = True

        return (
            np.array(observations, dtype=np.float32),
            np.array(rewards),
            dones,
            np.array(actions),
        )

    def action(self, state, stochastic=True):
        ob = np.asarray([state])
        dis = self.model.predict(ob)
        print(dis)
        if stochastic:
            act = np.random.choice(6, 1, p=dis[0])
            return ["U", "R", "D", "L", "I", "C"][act.item()]
        return ["U", "R", "D", "L", "I", "C"][np.argmax(dis)]
