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
    save_bc_model(model_dir, model)

    return model
    
    
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


def save_bc_model(model_dir, model):
    """
    Saves the specified model under the directory model_dir. This creates three items

        assets/         stores information essential to reconstructing the context and tf graph
        variables/      stores the model's trainable weights
        saved_model.pd  the saved state of the model object
    """   
    print("Saving bc model at ", model_dir)
    model.save(model_dir, save_format='tf')


def load_bc_model(model_dir):
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

class BC_policy:
    
    def __init__(self, dir = './bc_run', eager = False):
           
        # Save the session that the model was loaded into so it is available at inference time if necessary
        self._sess = get_session()
        self.eager = eager
        self.context = self._create_execution_context()
        self.action_shape = (len(Action.ALL_ACTIONS), )
        self.model = load_bc_model(dir)

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


class NNAgent(Agent):
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

def test(env, filename, nb_game=1, render=False):
    a0 = NNAgent(env, 400, 0)
    a1 = NNAgent(env, 400, 1)
    agent_pair = AgentPair(a0, a1)
    for i in range(nb_game):
        trajectories = env.get_rollouts(agent_pair, 1)
        
        trajectories = AgentEvaluator.make_trajectories_json_serializable(trajectories)
        with open("trajs/" + filename + str(i) + ".json", "w") as f:
            json.dump(traj2demo(trajectories), f)
                

if __name__ == "__main__":
    
    # _build_model([0], [6], MLP_PARAMS)
    model = train_bc_model('./bc_run', True)
    # bc_policy = BC_policy(eager = True)
    # inputs = [0, 1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -2, 2, 0, 0, 0, 2, 1, 0, 1, 0, 2, -1, 1, 2]
    # loss_less = np.zeros((1,5,4,26))
    # print(np.shape(np.asarray([inputs])))
    # print(np.shape(loss_less))
    # print(bc_policy.action([np.asarray([inputs]),loss_less]))
    # load_from_json()
    
    # recontructed_model, _ = load_bc_model('./bc_run')
    # print(np.argmax(recontructed_model.predict([np.asarray([inputs]),loss_less])))
    
    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
    test(env.base_env, "cnn_mlp", 1)
    