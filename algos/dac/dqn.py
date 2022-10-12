import os
from madacbench.mamo.saenv import MOEAEnv
import argparse
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='WFG6_3')
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    gamma = 0.99
    n_step = 3
    target_update_freq = 160
    buffer_size = 50000
    eps_train = 0.1
    epoch = 20
    step_per_epoch = 20000
    step_per_collect = 32
    update_per_step = 0.05
    batch_size = 32
    episode_per_test = 15


if __name__ == "__main__":
    args = get_args()
    args.early_stop = True
    args_test = get_args()
    dqn_args = DQNargs()
    env = MOEAEnv(**vars(args))
    train_envs = SubprocVectorEnv(
        [lambda: MOEAEnv(**vars(args)) for _ in range(dqn_args.train_num)])
    args_test.key = "WFG6" + args.key[-2:]
    test_envs = SubprocVectorEnv(
        [lambda: MOEAEnv(**vars(args_test)) for _ in range(dqn_args.test_num)])
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape, hidden_sizes=dqn_args.hidden_sizes,
              device=dqn_args.device).to(dqn_args.device)
    optim = torch.optim.Adam(net.parameters(), lr=dqn_args.lr)
    policy = DQNPolicy(net, optim, discount_factor=dqn_args.gamma,
                       estimation_step=dqn_args.n_step, target_update_freq=dqn_args.target_update_freq)
    buf = VectorReplayBuffer(dqn_args.buffer_size, buffer_num=len(train_envs))
    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    log_path = os.path.join('./results/dqn', args.key)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(
            log_path, args.key + 'policy.pth'))

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 100000:
            policy.set_eps(dqn_args.eps_train)
        elif env_step <= 500000:
            eps = dqn_args.eps_train - (env_step - 100000) / \
                400000 * (0.9 * dqn_args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * dqn_args.eps_train)

    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=dqn_args.epoch,
        step_per_epoch=dqn_args.step_per_epoch,
        step_per_collect=dqn_args.step_per_collect,
        update_per_step=dqn_args.update_per_step,
        episode_per_test=dqn_args.episode_per_test,
        batch_size=dqn_args.batch_size,
        train_fn=train_fn,
        save_fn=save_best_fn,
        logger=logger
    )
    print(f'Finished training! Use {result["duration"]}')
