from abc import ABC, abstractmethod


class MultiAgentEnv(ABC):
    @abstractmethod
    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    @abstractmethod
    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    @abstractmethod
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    @abstractmethod
    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    @abstractmethod
    def get_avail_actions(self):
        raise NotImplementedError

    @abstractmethod
    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    @abstractmethod
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def seed(self):
        raise NotImplementedError

    @abstractmethod
    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
