import argparse
import logging
import gym
import numpy as np
import tensorflow as tf
import overcooked_gym_env
from overcooked_ai_py.mdp.actions import Action
from il_model_utils import load_from_json, NullContextManager, TfContextManager, build_model, save_model
from traj_agent_ac import A2CAgent, Model

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.95)
    # parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()

class Discriminator:
    def __init__(self, env):
        self.env = env
    

def train(args):
    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
    model = Model(num_actions=len(Action.ALL_ACTIONS))
    agent = A2CAgent(gamma=args.gamma)
    D = Discriminator(env)
    obs = env.reset()
    success_num = 0
    #load expert observations and actions
    expert_observations, expert_observations_from_pixels, expert_actions = load_from_json()

    for iterations in range(args.iteration):
        observations = []
        actions = []
        rewards = []
        v_preds = []
        dones = []
        ep_rewards = [0.0]
        run_policy_steps = 0
        while True:
            run_policy_steps += 1
            act, v_pred = agent.model.action_value(obs[None, :])
            act = np.asscalar(act)
            v_pred = np.asscalar(v_pred)

            next_obs,reward,done,info = env.step(act)

            observations.append(obs)
            actions.append(act)
            rewards.append(reward)
            v_preds.append(v_pred)
            dones.append(done)

            if done:
                agent.nb_episodes += 1
                ep_rewards.append(0.0)
                next_obs = env.reset()
                logging.info("Episode: %03d, Reward: %.2f" % (len(ep_rewards) - 1, ep_rewards[-2]))
                break
            else:
                obs = next_obs


        if sum(rewards) >= 195:
            success_num += 1
            if success_num >= 25:
                agent.save_model()
                print('Clear!! Model saved.')
                break
        else:
            success_num = 0

        observations = np.asarray(observations)
        # observations = np.reshape(observations,newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype = np.int32)

        for i in range(2):
            D.train(expert_s = expert_observations,
                    expert_a = expert_actions,
                    agent_s = observations,
                    agent_a = actions)


        d_rewards = D.get_rewards(agent_s=observations,agent_a = actions)
        d_rewards = np.reshape(d_rewards,newshape=[-1]).astype(dtype=np.float32)

        _, next_value = agent.model.action_value(next_obs[None, :])
        returns, advs = agent._returns_advantages_rl(rewards, dones, v_preds, next_value)
        # A trick to input actions and advantages through same API.
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        

        # train policy
        for epoch in range(5):
            losses = agent.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (iterations + 1, args.iteration, losses))


if __name__ == '__main__':
    args = argparser()
    train(args)