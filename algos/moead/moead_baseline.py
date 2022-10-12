from madacbench.mamo.maenv import MOEAEnv
from madacbench.mamo.mamo_register import Task
from madacbench.mamo.agents import *
import argparse
import numpy as np
import random
import ray
import os

task = Task.get_task(name="all")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='WFG6_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--action', nargs="*", type=int, default=[1, 0, 1, 0])
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--save-history', action="store_true", default=False)
    parser.add_argument('--wo-obs', action="store_true",
                        default=False)  # use adaptive Weights
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--description', type=str, default='test')
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_known_args()[0]
    return args


@ray.remote
def step_in_env(args):
    env = MOEAEnv(key=args.key, population_size=args.population_size, n_ref_points=args.n_ref_points,
                  budget_ratio=args.budget_ratio, save_history=args.save_history, early_stop=args.early_stop,
                  baseline=True, wo_obs=args.wo_obs, test=args.test)
    env.reset()
    info = {'best_igd': 1e6,
            'last_igd': 1e6}
    is_ter = False
    while not is_ter:
        _, is_ter, info = env.step(args.action)
    return info


def moea_run_baseline(args=get_args()):
    args.save_history = True
    args.wo_obs = True
    args.action[0] = 1
    args.action[2] = 1
    save_path = './results/moead/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    info = ray.get([step_in_env.remote(args) for _ in range(args.repeat)])
    action = "baseline"
    action += "_Ne" + str(neighbor_size(args.action[0]))
    np.savez(
        f'{save_path}{args.key}_{action}_sd{args.seed}_rp{args.repeat}.npz',
        info_stack=info)


if __name__ == '__main__':
    args = get_args()
    ray.init(num_cpus=args.repeat)
    for t in task:
        args.key = t
        moea_run_baseline(args)
        print("===== Finish" + args.key + " =====")
