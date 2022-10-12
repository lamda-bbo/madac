from abc import abstractmethod
import csv
import logging
import numpy as np

from gym import spaces
from madacbench.multiagentenv import MultiAgentEnv


class SigmoidBase(MultiAgentEnv):
    """
    SigmoidMultiActMultiValAction
    Each agent has the state space only for itself
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self,
                 n_steps: int = 10,
                 n_agents: int = 5,  # agents number
                 n_actions: int = 3,  # action each agent
                 action_vals: tuple = (5, 10),
                 seed: bool = 0,
                 noise: float = 0.0,
                 instance_feats: str = None,
                 slope_multiplier: float = 2,
                 key: str = "Sigmoid",
                 replay_dir=None
                 ) -> None:
        super().__init__()
        action_vals = [n_actions for i in range(n_agents)]
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        assert self.n_agents == len(action_vals), (
            f'action_vals should be of length {self.n_agents}.')
        self.n_actions = n_actions
        self.shifts = [self.n_steps / 2 for _ in action_vals]
        self.slopes = [-1 for _ in action_vals]
        self.reward_range = (0, 1)
        self._c_step = 0
        self.noise = noise
        self.slope_multiplier = slope_multiplier
        self.action_vals = action_vals
        # budget spent, inst_feat_1, inst_feat_2
        # self._state = [-1 for _ in range(3)]
        # self.action_space = spaces.MultiDiscrete(action_vals)
        self.action_space = spaces.Discrete(int(np.prod(action_vals)))
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for _ in range(1 + n_agents * 3)]),
            high=np.array([np.inf for _ in range(1 + n_agents * 3)]))

        # initial State and Obs
        self._c_step = 0
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget]
        self.obs = []
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
            self.obs.append([remaining_budget, shift, slope, -1])
        next_state += [-1 for _ in range(self.n_agents)]
        self.state = next_state

        self.logger = logging.getLogger(self.__str__())
        self.logger.setLevel(logging.ERROR)
        self._prev_state = None
        self._inst_feat_dict = {}
        self._inst_id = None
        self.seed = 0
        if instance_feats:
            with open(instance_feats, 'r') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self._inst_feat_dict[int(row['ID'])] = [float(shift) for shift in row['shifts'].split(
                        ",")] + [float(slope) for slope in row['slopes'].split(",")]
                self._inst_id = -1

        # For compatibility With Epymarl
        self.episode_limit = self.n_steps

    def step(self, action):
        """

        @param action:  List: [x,x,...,x]
        @return: Returns reward, terminated, info
        """
        val = self._c_step
        action = action.cpu().tolist()
        r = [1 - np.abs(self._sig(val, slope, shift) - (act / (max_act - 1)))
             for slope, shift, act, max_act in zip(
            self.slopes, self.shifts, action, self.action_vals
        )]
        r = np.clip(np.prod(r), 0.0, 1.0)
        remaining_budget = self.n_steps - self._c_step

        next_state = [remaining_budget]
        self.obs = []
        for shift, slope, a in zip(self.shifts, self.slopes, action):
            next_state.append(shift)
            next_state.append(slope)
            self.obs.append([remaining_budget, shift, slope, a])
        next_state += action
        prev_state = self._prev_state
        self.state = next_state

        self.logger.debug("i: (s, a, r, s') / %d: (%s, %s, %5.2f, %2s)", self._c_step - 1, str(prev_state),
                          str(action), r, str(next_state))
        self._c_step += 1
        self._prev_state = next_state
        return r, self._c_step >= self.n_steps, {}

    @abstractmethod
    def reset(self):
        pass

    def render(self, mode: str, close: bool = True) -> None:
        pass

    @abstractmethod
    def get_obs(self):
        """ Returns all agent observations in a list """
        pass

    @abstractmethod
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        pass

    @abstractmethod
    def get_obs_size(self):
        """ Returns the shape of the observation """
        pass

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1 for i in range(self.n_actions)]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def close(self):
        pass

    def seed(self):
        return self.seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return None
