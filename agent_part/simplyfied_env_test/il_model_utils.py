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
import overcooked_gym_env

MLP_PARAMS = {
    "num_layers" : 2,
    "net_arch" : [128, 64]
}

TRAINING_PARAMS = {
    "epochs" : 500,
    "validation_split" : 0.15,
    "batch_size" : 64,
    "learning_rate" : 1e-3,
    "use_class_weights" : False
}


def load_from_json(data_dir = ['./train_data/1.json','./train_data/2.json','./train_data/3.json','./train_data/4.json']):
    
    def joint_state_trajectory_to_single(trajectories, traj, idx):
        dummy_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        dummy_env = OvercookedEnv.from_mdp(dummy_mdp)
        states, joint_actions = traj['ep_observations'][0], traj['ep_actions'][0]
        for agent_idx in idx:
            ep_obs, ep_acts, ep_lossless = [], [], []
            for i in range(len(states)):
                state_d = states[i]
                for player_dict in state_d['players']:
                    if not 'held_object' in player_dict:
                        player_dict['held_object'] = None
                state, action = OvercookedState.from_dict(state_d), joint_actions[i][agent_idx]
                if type(action) != str:
                    action = tuple(action)
                elif action == "INTERACT":
                    action = 'interact'
                else:
                    action = 'stay'
                action = np.array([Action.ACTION_TO_INDEX[action]]).astype(int)
                estate = dummy_env.featurize_state_mdp(state)[agent_idx]
                lossless = dummy_env.lossless_state_encoding_mdp(state)[agent_idx]
                ep_obs.append(estate)
                ep_acts.append(action)
                ep_lossless.append(lossless)
        trajectories["ep_observations"].append(ep_obs)
        trajectories["ep_actions"].append(ep_acts)
        trajectories["ep_lossless"].append(ep_lossless)
    
    
    def process_traj(traj):
        single_agent_trajectories = {
        # With shape (n_timesteps, game_len), where game_len might vary across games:
        "ep_observations": [],
        "ep_actions": [],
        "ep_lossless": [],
        }
        joint_state_trajectory_to_single(single_agent_trajectories, traj, [0, 1])
        return single_agent_trajectories
    
    inputs = []
    img = []
    targets = []
    for f in data_dir:
        with open(f, 'r') as fp:
            traj = json.load(fp)
            processed_trajs = process_traj(traj)
            inputs.extend(processed_trajs["ep_observations"])
            targets.extend(processed_trajs["ep_actions"])
            img.extend(processed_trajs["ep_lossless"])
    inputs = np.vstack(inputs)
    img = np.vstack(img)
    targets = np.vstack(targets)
    return inputs, img, targets

    
def build_model(input_shape, action_shape, eager = False, **kwargs):
    
    ## observation Inputs
    inputs = keras.Input(shape=input_shape[0], name="Overcooked_observation")
    x = inputs
    for i in range(MLP_PARAMS["num_layers"]):
        units = MLP_PARAMS["net_arch"][i]
        x = keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i))(x)
        
    #lossless inputs
    lossless_inputs = keras.Input(shape=input_shape[1], name="Overcooked_lossless")
    conv_1 = keras.layers.Conv2D(32, 2, activation='relu', padding="same")(lossless_inputs)
    maxpool_1 = keras.layers.MaxPooling2D(padding="same")(conv_1)
    conv_2 = keras.layers.Conv2D(24, 2, activation='relu', padding="same")(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D(padding="same")(conv_2)
    conv_3 = keras.layers.Conv2D(16, 2, activation='relu', padding="same")(maxpool_2)
    cnn = keras.layers.Flatten()(conv_3)    
        
    x = keras.layers.concatenate([cnn, x])    
    ## output layer
    logits = keras.layers.Dense(action_shape[0], name="logits")(x)
    return keras.Model(inputs=[inputs, lossless_inputs], outputs=logits)


def Discriminator(OvercookedEnv):
    pass


def save_model(model_dir, model):
    """
    Saves the specified model under the directory model_dir. This creates three items

        assets/         stores information essential to reconstructing the context and tf graph
        variables/      stores the model's trainable weights
        saved_model.pd  the saved state of the model object
    """   
    print("Saving bc model at ", model_dir)
    model.save(model_dir, save_format='tf')


def load_model(model_dir):
    """
    Returns the model instance (including all compilation data like optimizer state) and a dictionary of parameters
    used to create the model
    """
    print("Loading bc model from ", model_dir)
    model = keras.models.load_model(model_dir, custom_objects={ 'tf' : tf })
    return model


class NullContextManager:
    """
    No-op context manager that does nothing
    """
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass


class TfContextManager:
    """
    Properly sets the execution graph and session of the keras backend given a "session" object as input

    Used for isolating tf execution in graph mode. Do not use with eager models or with eager mode on
    """
    def __init__(self, session):
        self.session = session
    def __enter__(self):
        self.ctx = self.session.graph.as_default()
        self.ctx.__enter__()
        set_session(self.session)
    def __exit__(self, *args):
        self.ctx.__exit__(*args)
