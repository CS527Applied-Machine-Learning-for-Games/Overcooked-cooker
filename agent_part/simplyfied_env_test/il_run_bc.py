import argparse
import os, pickle, copy, json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session, get_session

from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, Agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from human_aware_rl.static import HUMAN_DATA_PATH
from human_aware_rl.human.process_dataframes import get_trajs_from_data

from traj_agent import traj2demo
from il_model_utils import NullContextManager, TfContextManager, build_model, save_model, load_model, load_from_json, TRAINING_PARAMS
import overcooked_gym_env


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train/test', default='test')
    return parser.parse_args()


class BC_policy:
    
    def __init__(self, dir = './bc_run', eager = False):
           
        # Save the session that the model was loaded into so it is available at inference time if necessary
        self._sess = get_session()
        self.eager = eager
        self.context = self._create_execution_context()
        self.action_shape = (len(Action.ALL_ACTIONS), )
        self.model = load_model(dir)

    def action(self, obs_batch):
        with self.context:
            action_logits = self._forward(obs_batch)
        
        def softmax(logits):
                e_x = np.exp(logits.T - np.max(logits))
                return (e_x / np.sum(e_x, axis=0)).T
        action_probs = softmax(action_logits)
        action = np.array([np.random.choice(self.action_shape[0], p=action_probs[i]) for i in range(len(action_probs))])
        action = Action.INDEX_TO_ACTION[action[0]]
        return action,  { "action_dist_inputs" : action_logits }

    def _forward(self, obs_batch):
        return self.model.predict(obs_batch)

    def _create_execution_context(self):
        """
        Creates a private execution context for the model 

        Necessary if using with rllib in order to isolate this policy model from others
        """
        if self.eager:
            return NullContextManager()
        return TfContextManager(self._sess)


class BCAgent(Agent):
    
    def __init__(self, env,  horizon, idx):
        super().__init__()
        self.env = env
        self.horizon = horizon
        self.policy = BC_policy(eager = True)
        self.idx = idx

    def action(self, state):
        estate = self.env.featurize_state_mdp(state)[self.idx]
        obs = self.env.lossless_state_encoding_mdp(state)[self.idx]
        action = self.policy.action([np.asarray([estate]), np.asarray([obs])])
        return action


def train_bc_model(model_dir, verbose = False):
    inputs, img, targets = load_from_json()
    print('input shape:', np.shape(inputs))
    print('img shape:', np.shape(img))

    # Retrieve un-initialized keras model
    model = build_model([np.shape(inputs)[1:], np.shape(img)[1:]], (len(Action.ALL_ACTIONS), ))

    # Initialize the model
    # Note: have to use lists for multi-output model support and not dicts because of tensorlfow 2.0.0 bug
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["sparse_categorical_accuracy"]
    model.compile(optimizer=keras.optimizers.Adam(TRAINING_PARAMS["learning_rate"]),
                  loss=loss,
                  metrics=metrics)


    # Create input dict for both models
    N = inputs.shape[0]
    inputs = { "Overcooked_observation" : inputs , "Overcooked_lossless": img} 
    targets = { "logits" : targets }

    # Batch size doesn't include time dimension (seq_len) so it should be smaller for rnn model
    batch_size = TRAINING_PARAMS['batch_size']
    model.fit(inputs, targets, callbacks=None, batch_size=batch_size, 
                epochs=TRAINING_PARAMS['epochs'], validation_split=TRAINING_PARAMS["validation_split"],
                verbose=2 if verbose else 0)

    # # Save the model
    save_model(model_dir, model)

    return model


def test_bc_agent(env, filename, nb_game=1, render=False):
    a0 = BCAgent(env, 400, 0)
    a1 = BCAgent(env, 400, 1)
    agent_pair = AgentPair(a0, a1)
    for i in range(nb_game):
        trajectories = env.get_rollouts(agent_pair, 1)
        
        trajectories = AgentEvaluator.make_trajectories_json_serializable(trajectories)
        with open("trajs/" + filename + str(i) + ".json", "w") as f:
            json.dump(traj2demo(trajectories), f)
                

if __name__ == "__main__":
    args = argparser()
    if args.mode ==  'train':
        model = train_bc_model('./bc_run', True)
    else:
        env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
        test_bc_agent(env.base_env, "cnn_mlp", 1)
    