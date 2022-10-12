from madacbench.sigmoid.base_env import SigmoidBase


class SigmoidMultiActMultiValAction(SigmoidBase):
    """
    Each agent has the state space only for itself
    """

    def reset(self):
        """ Returns initial observations and states"""
        if self._inst_feat_dict:
            self._inst_id = (self._inst_id + 1) % len(self._inst_feat_dict)
            self.shifts = self._inst_feat_dict[self._inst_id][:self.n_agents]
            self.slopes = self._inst_feat_dict[self._inst_id][self.n_agents:]
        else:
            self.shifts = self.rng.normal(
                self.n_steps / 2, self.n_steps / 4, self.n_agents)
            self.slopes = self.rng.choice([-1, 1], self.n_agents) * self.rng.uniform(
                size=self.n_agents) * self.slope_multiplier
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
        self._prev_state = None
        self.logger.debug(
            "i: (s, a, r, s') / %d: (%2d, %d, %5.2f, %2d)", -1, -1, -1, -1, -1)
        return self.obs, self.state

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.obs[0])
