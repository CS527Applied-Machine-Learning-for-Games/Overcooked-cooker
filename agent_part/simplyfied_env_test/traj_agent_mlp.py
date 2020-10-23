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
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, \
    SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS, Overcooked
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, ONION_DISPENSER, TOMATO_DISPENSER, POT, \
    DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent, Agent, \
    StayAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir, load_from_json, \
    save_as_json

import itertools
import json
from collections import defaultdict, Counter
import gym

import overcooked_gym_env

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=1200)
parser.add_argument('-n', '--num_updates', type=int, default=2500)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-6)
parser.add_argument('-ga', '--gamma', type=float, default=0.99)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)


def traj2demo(trajectories):
    trajectories["ep_observations"] = trajectories.pop("ep_states")
    for ep_list in trajectories["ep_observations"]:
        for step_dicts in ep_list:
            step_dicts["order_list"] = None
            for obj in step_dicts["objects"]:
                if obj["name"] == "soup":
                    obj["state"] = [obj["_ingredients"][0]["name"],
                                    len(obj["_ingredients"]),
                                    max([0, obj["cooking_tick"]])]
            for player_dict in step_dicts["players"]:
                held_obj = player_dict["held_object"]
                if held_obj is not None and held_obj["name"] == "soup":
                    held_obj["state"] = [held_obj["_ingredients"][0]["name"],
                                         len(held_obj["_ingredients"]),
                                         held_obj["cooking_tick"]]

    for ep_list in trajectories["ep_actions"]:
        for i in range(len(ep_list)):
            ep_list[i] = list(ep_list[i])
            for j in range(len(ep_list[i])):
                if isinstance(ep_list[i][j], str):
                    ep_list[i][j] = 'INTERACT'

    for ep_params in trajectories["mdp_params"]:
        extra_info = {
            "num_items_for_soup": 3,
            "rew_shaping_params": None,
            "cook_time": 20,
            "start_order_list": None
        }
        ep_params.update(extra_info)

    return trajectories


def demo2traj(filename):
    traj_dict = load_from_json(filename)
    for ep_list in traj_dict["ep_observations"]:
        for step_dict in ep_list:
            for state_dict in step_dict["players"]:
                if "held_object" not in state_dict:
                    state_dict["held_object"] = None

    traj_dict["ep_states"] = [[OvercookedState.from_dict(ob) for ob in curr_ep_obs] for curr_ep_obs in
                              traj_dict["ep_observations"]]
    traj_dict["ep_actions"] = [[tuple(tuple(a) if type(a) is list else "interact" for a in j_a) for j_a in ep_acts] for
                               ep_acts
                               in traj_dict["ep_actions"]]
    return traj_dict


class NNAgent(Agent):
    def __init__(self, env, model, horizon):
        super().__init__()
        self.env = env
        self.model = model
        self.horizon = horizon

    def action(self, state):
        obs = self.env.featurize_state_mdp(state)
        # action, value = self.model.action_value(obs[0][None, :].astype(np.float32))
        logits = self.model.predict(obs[0][None, :].astype(np.float32))
        score = tf.nn.softmax(logits[0])
        # print(score)
        action = np.argwhere(np.random.multinomial(1, score) == 1)[0][0]
        # print(action)
        action = Action.INDEX_TO_ACTION[action]
        return action, {}


class A2CAgent:
    def __init__(self, lr=5e-5, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        print(self.model)
        print(type(self.model))

    def train(self, env, filename, batch_sz=1200, epochs=10):
        traj = demo2traj(filename)
        observations, rewards, dones, actions = self.traj2data(traj, env)
        # print(observations.shape)
        self.model.fit(observations, actions, epochs=epochs)

    def test(self, env, filename, nb_game=1, render=False):
        a0 = NNAgent(env, self.model, env.horizon)
        a1 = StayAgent()
        agent_pair = AgentPair(a0, a1)
        # trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        for i in range(nb_game):
            trajectories = env.get_rollouts(agent_pair, 1)
            trajectories = AgentEvaluator.make_trajectories_json_serializable(trajectories)

            with open("trajs/" + filename + str(i) + ".json", "w") as f:
                json.dump(traj2demo(trajectories), f)

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
                observation = env.featurize_state_mdp(ep_list[i])[0]
                observations.append(observation)

        for ep_list in trajectories["ep_actions"]:
            for j_a in ep_list:
                actions.append(action2index[j_a[0]])

        for ep_list in trajectories["ep_rewards"]:
            rewards += ep_list
            dones.append([False for _ in range(len(ep_list))])
            dones[-1] = True

        return np.array(observations, dtype=np.float32), np.array(rewards), dones, np.array(actions)


if __name__ == '__main__':
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
    agent = A2CAgent(args.learning_rate, gamma=args.gamma)

    agent.train(env.base_env, "trajs/10_10_2020_19_42_3_ppo_bc_1_long.json", epochs=20)
    agent.test(env.base_env, "simple_cross_entropy_mlp", 3)
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
