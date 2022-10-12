import os
from madacbench.mamo.saenv import MOEAEnv
from madacbench.mamo.mamo_register import Task
import argparse
import torch
import numpy as np
import random
import ray

from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from tianshou.data import Batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_5')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--save-history', action="store_true", default=False)
    parser.add_argument('--baseline', action="store_true",
                        default=False)  # use default operator Type
    parser.add_argument('--adaptive-open', action="store_true",
                        default=False)  # use adaptive Weights
    parser.add_argument('--wo-obs', action="store_true",
                        default=False)  # use adaptive Weights
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


class DQNargs:
    train_num = 16
    test_num = 4
    hidden_sizes = [128, 128, 128]
    lr = 1e-3
    gamma = 0.9
    n_step = 3
    target_update_freq = 160
    buffer_size = 20000
    eps_train = 0.1
    epoch = 20
    step_per_epoch = 5000
    step_per_collect = 32
    update_per_step = 0.05
    batch_size = 64
    episode_per_test = 15
    repeat = 30


def set_global_seeds(seed: int):
    """
    Set the random seed of pytorch, numpy and random.
    params:
        seed: an integer refers to the random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@ray.remote
def step_in_env(args, policy):
    env = MOEAEnv(**vars(args))
    obs = env.reset()

    info = {'best_igd': 1e6,
            'last_igd': 1e6}
    is_ter = False
    while not is_ter:
        batch = Batch(obs=[obs], info={})
        act = policy(batch).act[0]
        obs, r, is_ter, info = env.step(act)
    return info


def dqn_run_baseline(args=get_args(), dqn_args=None, policy_name=None):
    args.save_history = True
    save_path = './results/dqn/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = MOEAEnv(**vars(args))
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, hidden_sizes=dqn_args.hidden_sizes)
    optim = torch.optim.Adam(net.parameters(), lr=dqn_args.lr)
    policy = DQNPolicy(net, optim, discount_factor=dqn_args.gamma,
                       estimation_step=dqn_args.n_step, target_update_freq=dqn_args.target_update_freq)
    policy_weights = torch.load(
        f'./results/dqn/{policy_name}/{policy_name}policy.pth')
    policy.load_state_dict(policy_weights)
    policy.eval()

    info = ray.get([step_in_env.remote(args, policy)
                   for _ in range(dqn_args.repeat)])
    np.savez(
        f'{save_path}{args.key}_sd{args.seed}_rp{dqn_args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()
    policy_name = args.key
    dqn_args = DQNargs()
    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + args.key.split("_")[-1])
    ray.init(num_cpus=dqn_args.repeat)
    for t in task:
        args.key = t
        dqn_run_baseline(args, dqn_args, policy_name)
        print("===== Finish" + args.key + " =====")
