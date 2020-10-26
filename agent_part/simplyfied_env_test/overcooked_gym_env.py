from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS, Overcooked
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir, load_from_json, save_as_json

import numpy as np
import itertools
from collections import defaultdict, Counter
import gym


class MyOvercookedEnv(Overcooked):
    env_name = "Overcooked-v0"

    def custom_init(self, base_env, featurize_fn, baselines_reproducible=False):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)

        self.base_env = base_env
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def featurize_fn(self, mdp, state):
        return mdp.lossless_state_encoding(state, horizon=self.base_env.horizon)

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape
        high = np.ones(obs_shape) * 20
        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        # assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (action, type(action))
        # agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        # joint_action = (agent_action, other_agent_action)
        joint_action = (Action.INDEX_TO_ACTION[action], Action.STAY)

        next_state, reward, done, env_info = self.base_env.step(joint_action)

        ob_p0, ob_p1 = self.featurize_fn(self.mdp, next_state)

        # both_agents_ob = (ob_p0, ob_p1)

        env_info["policy_agent_idx"] = self.agent_idx

        if "episode" in env_info.keys():
            env_info["episode"]["policy_agent_idx"] = self.agent_idx

        # obs = {"both_agent_obs": both_agents_ob,
        #        "overcooked_state": next_state,
        #        "other_agent_env_idx": 1 - self.agent_idx}
        return ob_p0.astype(np.float32), reward, done, env_info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        self.agent_idx = 0
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, self.base_env.state)

        # both_agents_ob = (ob_p0, ob_p1)

        return ob_p0.astype(np.float32)

        # return {"both_agent_obs": both_agents_ob,
        #         "overcooked_state": self.base_env.state,
        #         "other_agent_env_idx": 1 - self.agent_idx}


def get_gym_env(layout_name="cramped_room", horizon=1000, params_to_overwrite=None):
    if params_to_overwrite:
        mdp = OvercookedGridworld.from_layout_name(layout_name, **params_to_overwrite)
    else:
        mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    my_env = MyOvercookedEnv()
    my_env.custom_init(env, None)
    return my_env

