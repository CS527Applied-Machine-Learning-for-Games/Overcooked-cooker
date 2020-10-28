import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
import overcooked_gym_env
from il_model_utils import load_from_json, NullContextManager, TfContextManager, build_model, save_model, Discriminator


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def train(args):
    env = overcooked_gym_env.get_gym_env(layout_name="cramped_room", horizon=400)
    Policy = build_model()
    Old_Policy = build_model()
    PPO = None # PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(env)
    success_num = 0
    #load expert observations and actions
    expert_observations, expert_observations_from_pixels, expert_actions = load_from_json()

    for iterations in range(args.iteration):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            
            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                next_obs,reward,done,info = env.step(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs = env.reset()
                    break
                else:
                    obs = next_obs


            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 25:
                    save_model('gail_run/success_{}'.format(success_num), Policy)
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            observations = np.reshape(observations,newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype = np.int32)

            for i in range(2):
                D.train(expert_s = expert_observations,
                        expert_a = expert_actions,
                        agent_s = observations,
                        agent_a = actions)


            d_rewards = D.get_rewards(agent_s=observations,agent_a = actions)
            d_rewards = np.reshape(d_rewards,newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for epoch in range(6):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])


if __name__ == '__main__':
    args = argparser()
    train(args)