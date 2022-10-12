import copy
import random
import numpy as np

from madacbench.multiagentenv import MultiAgentEnv
from madacbench.mamo.moead_env import MamoBase


class MOEAEnv(MultiAgentEnv):
    def __init__(self,
                 key="WFG6_3",
                 n_ref_points=1000,
                 population_size=210,
                 seed=2022,
                 budget_ratio=50,
                 test=False,
                 wo_obs=False,
                 save_history=False,
                 early_stop=True,
                 baseline=False,
                 adaptive_open=False,
                 replay=False,
                 replay_dir=None,
                 ban_agent=4,
                 reward_type=0):
        super().__init__()
        self.env = MamoBase(key,
                            n_ref_points,
                            population_size,
                            seed,
                            budget_ratio,
                            test,
                            wo_obs,
                            save_history,
                            early_stop,
                            baseline,
                            adaptive_open,
                            replay,
                            replay_dir,
                            ban_agent,
                            reward_type)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env.episode_limit

    def step(self, action):
        """
        Returns reward, terminated, info
        :type action: int or np.int
        :param action: for dimension: Neighbor Size, Operator Type, Operator Parameter, Adaptive Weights
                [x, x, x, x]: shape=n_agents
        :return: obs, reward, is_done, info
        """

        reward, done, info = self.env.moead_step(action)
        return reward, done, info

    def reset(self):
        """ Returns initial observations and states """
        self.env.moead_reset()
        return self.env.obs

    def close(self):
        self.reset()

    def get_obs(self):
        """ Returns all agent observations in a list """
        return copy.deepcopy([self.env.obs for _ in range(self.env.n_agents)])

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return copy.deepcopy(self.env.obs)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.n_obs

    def get_state(self):
        return copy.deepcopy(self.env.obs)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.env.n_obs

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        if self.env.ban_agent != 3:
            if agent_id < 3:  # 0,1,2
                if agent_id == self.env.ban_agent:
                    return [0, 1, 0, 0]
                else:
                    return [1, 1, 1, 1]
            else:  # 3
                if self.env.adaptive_cooling_time > 0 or (
                        np.sum(self.env.action_count[0]) >= self.env.adaptive_end // self.env.population_size):
                    return [1, 0, 0, 0]
                else:
                    return [1, 1, 0, 0]
        elif self.env.ban_agent == 3:
            if agent_id < 3:
                return [1, 1, 1, 1]
            else:
                return [1, 0, 0, 0]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.env.n_actions

    def render(self):
        raise NotImplementedError

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_stats(self):
        return None

    def save_replay(self):
        return self.env.save_replay()


if __name__ == "__main__":
    env = MOEAEnv()
    env.reset()
    for i in range(100):
        action = np.hstack([np.random.randint(0, 4, 3),
                           np.random.randint(0, 2, 1)])
        reward, done, info = env.step(action)
        print(reward, done, info)
    env.close()
