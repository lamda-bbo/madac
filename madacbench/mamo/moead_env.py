from madacbench.mamo.mamo_register import get_maenv
from madacbench.mamo.agents import operator_type, operator_parameter, neighbor_size
from madacbench.mamo import rmoea_problems

import copy
import random
import sys
from operator import itemgetter

import scipy
import tianshou
from platypus.config import PlatypusConfig
from platypus.core import Archive
from platypus.evaluator import Job
from platypus.indicators import *
from platypus.operators import TournamentSelector, RandomGenerator, \
    GAOperator, SBX, PM
from platypus.weights import random_weights, chebyshev
from pymoo.factory import get_reference_directions
import numpy as np
import os

EPSILON = sys.float_info.epsilon


class _EvaluateJob(Job):
    def __init__(self, solution):
        super(_EvaluateJob, self).__init__()
        self.solution = solution

    def run(self):
        self.solution.evaluate()


def get_weights(n_objs, n_points):
    tmp_weights = get_reference_directions(
        "das-dennis", n_objs, n_points=n_points).tolist()
    if len(tmp_weights) != n_points:
        print(
            f'Generate {len(tmp_weights)} weights but asked to generate {n_points}')
    return tmp_weights


def choose_problems(env_name, n_obj):
    return getattr(rmoea_problems, env_name)(n_obj)


def get_variable_boundary(problem):
    nvars = problem.nvars
    lb = np.zeros(shape=nvars)
    ub = np.zeros(shape=nvars)
    for i in range(nvars):
        lb[i] = problem.types[i].min_value
        ub[i] = problem.types[i].max_value
    return lb, ub


class MamoBase:
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
        """

        @param key: Task Name
        @param n_ref_points:
        @param population_size:
        @param seed:
        @param budget_ratio:
        @param test:
        @param wo_obs:
        @param save_history: Save History in return Info
        @param early_stop:
        @param baseline: Use Baseline Operator
        @param adaptive_open: Use Adaptive Weights
        @param replay: Save Replay in Replay dir
        @param replay_dir:
        @param ban_agent: 0,1,2,3 , if not , means no ban
        @param reward_type: 0 is Triangles, 1,2,3 is defined in DEDDQN
        """
        # Environment Change
        self.key = key
        self.func_choice = get_maenv(key)[0]
        self.nobjs_choice = get_maenv(key)[1]
        self.func_select = [
            (func, nobjs) for func in self.func_choice for nobjs in self.nobjs_choice]
        self.fun_index = 0
        random.shuffle(self.func_select)
        self.mixture = False
        if len(key.split("_")[-1]) > 1:
            self.mixture = True  # Multi-nobjs training
        # Problem Related
        self.n_ref_points = n_ref_points
        self.population_size = population_size
        self.budget_ratio = budget_ratio
        self._init_problem()
        self.ban_agent = ban_agent
        self.reward_type = reward_type
        # Adaptive Weights Agent Related
        self.adaptive_open = adaptive_open
        self._init_adaptive_weights()
        # MDP Related
        self.n_agents = 4
        self.n_actions = 4  # each agent has total n_actions
        self.total_n_actions = 4 * 4 * 4 * 2
        self.n_obs = 22
        if self.mixture is False:
            self.episode_limit = budget_ratio * self.n_objs + 1
        else:
            self.episode_limit = budget_ratio * 7 + 1
        self.early_stop = early_stop
        self.obs = np.zeros(self.n_obs)
        self.wo_obs = wo_obs  # Without Obs, False means that we need obs
        # MOEA/D Algorithm Related
        self.generator = RandomGenerator()
        self.selector = TournamentSelector(2)
        self.variator = None  # if variator is none, it will use the default variator
        self.evaluator = PlatypusConfig.default_evaluator
        self.baseline = baseline  # Used MOEA/D Default Operator Type
        self.moead_neighborhood_size = 30
        self.moead_neighborhood_maxsize = 30
        self.moead_eta = 2  # maximum number of replaced parent
        self.moead_delta = 0.8  # Use Neighbor or Whole Population, Important Parameters
        self.moead_weight_generator = random_weights
        # Not Important
        self.replay = replay
        self.replay_dir = replay_dir
        self.test = test
        self.save_history = save_history
        self.seed = seed
        self._init_static()

        self.replay_his = {
            "igd_his": [],
            "hv_his": [],
            "ndsort_ratio_his": [],
            "dis_his": [],
            "action_his": [],
            "population_his": []
        }

    def _init_problem(self):
        self.env_name = self.func_select[self.fun_index][0]
        self.n_objs = self.func_select[self.fun_index][1]
        self.problem = choose_problems(self.env_name, self.n_objs)
        self.lb, self.ub = get_variable_boundary(self.problem)
        self.problem_ref_points = self.problem.get_ref_set(
            n_ref_points=self.n_ref_points)
        self.igd_calculator = InvertedGenerationalDistance(
            reference_set=self.problem_ref_points)
        self.budget = self.population_size * self.budget_ratio * self.n_objs

    def _init_adaptive_weights(self):
        self.EP = []
        self.EP_MaxSize = int(self.population_size * 1.5)
        self.rate_update_weight = 0.05  # rate_update_weight * N = nus
        self.nus = int(
            self.rate_update_weight * self.population_size)  # maximal number of subproblems needed to be adjusted
        # adaptive iteration interval, Units are population
        self.wag = int(self.budget // self.population_size * 0.05)
        self.adaptive_cooling_time = self.wag
        self.adaptive_end = int(self.budget * 0.9)

    def _init_static(self):
        self.nfe = 0  # Discarded
        self.moead_generation = 0  # Discarded
        self.best_value = 1e6
        self.last_value = 1e6
        self.inital_value = None
        self.last_bonus = 0
        # The number of iterations without promotion, maximum 10 is stag_count_max
        self.stag_count = 0
        self.stag_count_max = (self.budget // self.population_size) / 10

        self.hv_his = []
        self.hv_last5 = tianshou.utils.MovAvg(size=5)

        self.nds_ratio_his = []
        self.nds_ratio_last5 = tianshou.utils.MovAvg(size=5)

        self.ava_dist_his = []
        self.ava_dist_last5 = tianshou.utils.MovAvg(size=5)

        self.value_his = []

        self.hv_running = tianshou.utils.RunningMeanStd()
        self.nds_ratio_running = tianshou.utils.RunningMeanStd()
        self.ava_dist_running = tianshou.utils.RunningMeanStd()

        self.action_count = np.zeros(
            shape=(self.n_agents, self.n_actions), dtype=int)
        self.action_seq_his = []  # record the sequential history of action

        self.info_reward_his = []
        self.info_obs_his = []
        self.info_igd_his = []

    def _update_obs(self):
        obs_ = np.zeros(self.n_obs)
        if not self.wo_obs:
            obs_[0] = 1 / self.problem.nobjs
            obs_[1] = 1 / self.problem.nvars
            obs_[2] = (np.sum(self.action_count[0]) *
                       self.population_size) / self.budget
            obs_[3] = self.stag_count / self.stag_count_max
            obs_[4] = self.get_hypervolume()
            obs_[5] = self.get_ratio_nondom_sol()
            obs_[6] = self.get_average_dist()
            obs_[7] = self.get_pre_k_change(1, self.hv_his)
            obs_[8] = self.get_pre_k_change(1, self.nds_ratio_his)
            obs_[9] = self.get_pre_k_change(1, self.ava_dist_his)
            obs_[10] = self.hv_last5.mean()
            obs_[11] = self.nds_ratio_last5.mean()
            obs_[12] = self.ava_dist_last5.mean()
            obs_[13] = self.hv_last5.std()
            obs_[14] = self.nds_ratio_last5.std()
            obs_[15] = self.ava_dist_last5.std()
            obs_[16] = self.hv_running.mean
            obs_[17] = self.nds_ratio_running.mean
            obs_[18] = self.ava_dist_running.mean
            obs_[19] = self.hv_running.var
            obs_[20] = self.nds_ratio_running.var
            obs_[21] = self.ava_dist_running.var
            return obs_
        else:
            return obs_

    def moead_initial(self):
        self.weights = get_weights(self.n_objs, self.population_size)
        self.neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(self.population_size):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            self.neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])
            self.moead_update_ideal(self.population[i])

    def moead_step(self, action=None):
        """
        one step update in moea/d
        inclue solution generation and solution selection
        @param action: neighboor size; operator type; operator parameter
        :return:
        """
        for id in range(self.n_agents):
            self.action_count[id][action[id]
                                  ] = self.action_count[id][action[id]] + 1
        self.action_seq_his.append(action)
        if action is None:
            raise Exception("Action Is None.")
        if self.adaptive_open is False:
            action[3] = 0
        self.moead_neighborhood_size = neighbor_size(action[0])
        scale = operator_parameter(action[2])  # Operator Parameter
        if np.sum(self.action_count) / self.n_agents <= 1:
            self.moead_initial()
        subproblems = self.moead_get_subproblems()
        self.offspring_list = []
        for index in subproblems:
            mating_indices = self.moead_get_mating_indices(index)
            # Generate an offspring
            if self.baseline is False:
                if index in mating_indices:
                    mating_indices.remove(index)
                de_pool = np.random.choice(mating_indices, 5, replace=False)
                offspring = [self.variator.mutation.evolve(
                    operator_type(action[1], self.population, index, de_pool, scale, self.lb, self.ub))]
            else:
                parents = [self.population[index]] + [self.population[i] for i in
                                                      np.random.permutation(mating_indices)[:(self.variator.arity - 1)]]
                offspring = self.variator.evolve(parents)
            self.evaluate_all(offspring)
            self.offspring_list.extend(offspring)
            for child in offspring:
                self.moead_update_ideal(child)
                self.moead_update_solution(child, mating_indices)  # selection
        if self.adaptive_open:
            self.update_ep()
        if action[3] > 1:
            raise Exception("action[3] > 1.")
        if action[3] == 1 and self.adaptive_cooling_time <= 0:
            self.adaptive_cooling_time = self.wag
            self.update_weight()
        self.moead_generation += 1
        self.adaptive_cooling_time -= 1
        value = self.get_igd()
        reward = self.get_reward(value)
        self.update_igd(value)
        self.obs = self._update_obs()
        if self.save_history:
            self.info_obs_his.append(self.obs)
        self.info_igd_his.append(self.last_value)  # Logger
        self.info_reward_his.append(reward)  # Logger
        if np.sum(self.action_count[0]) % 50 == 0 and self.test:
            # if test, it will report basic information of this run
            print("Problem {} ite {},  action is {}, best igd is {}, last igd is {}".format(self.problem,
                                                                                            self.nfe, action,
                                                                                            self.best_value,
                                                                                            self.last_value))
            print(self.action_count)
            print(self.obs)
        # if stop, then return the information
        if np.sum(self.action_count[0]) >= self.budget // self.population_size or \
                (self.stag_count > self.stag_count_max and self.early_stop):
            self.done = True
            if self.replay:
                self.update_replay()
        else:
            self.done = False
        info = {"best_igd": self.best_value, "last_igd": self.last_value}
        if self.save_history:
            info["igd_his"] = self.info_igd_his
            info["reward_his"] = self.info_reward_his
            info["obs_his"] = self.info_obs_his
        return reward, self.done, info

    def moead_reset(self):
        self.done = False
        self._init_problem()
        self._init_adaptive_weights()
        self._init_static()
        self.population = [self.generator.generate(
            self.problem) for _ in range(self.population_size)]
        self.evaluate_all(self.population)
        self.inital_value = self.get_igd()
        self.best_value = self.inital_value
        self.last_value = self.inital_value
        if self.n_objs < 5:
            tmp_population = [self.generator.generate(
                self.problem) for _ in range(int(1e5))]
            tmp_feasible = [
                s for s in tmp_population if s.constraint_violation == 0.0]
            self.evaluate_all(tmp_feasible)
            self.archive_maximum = (
                1.1 * np.array([max([s.objectives[i] for s in tmp_feasible]) for i in range(self.n_objs)])).tolist()
            del tmp_population, tmp_feasible
        else:
            self.archive_maximum = [
                max([s.objectives[i] for s in self.population]) for i in range(self.n_objs)]
        self.archive_minimum = [
            min([s.objectives[i] for s in self.population]) for i in range(self.n_objs)]
        self.ideal_point = copy.deepcopy(self.archive_minimum)
        if self.variator is None:
            self.variator = GAOperator(SBX(probability=1.0, distribution_index=20.0),
                                       PM(probability=1 / self.problem.nvars))
        self.obs = self._update_obs()
        # Change Task Function
        self.fun_index += 1
        if self.fun_index == len(self.func_select):
            self.fun_index = 0
            random.shuffle(self.func_select)

    def nondominated_solution(self, solutions):
        """
        Rank == 0 Indicates nondominated solution
        @param solutions:
        @return:
        """
        archive = Archive()  # An archive only containing non-dominated solutions.
        archive += solutions  # archive[0] is Solution Instance
        for solution in solutions:
            if solution in archive:
                solution.rank = 0
            else:
                solution.rank = 1

    def update_ep(self):
        """Update the current evolutional population EP
        """
        self.EP.extend(self.offspring_list)
        self.nondominated_solution(self.EP)
        self.EP = [e for e in self.EP if e.rank == 0]
        l = len(self.EP)
        if l <= self.EP_MaxSize:
            return
        # Delete the overcrowded solutions in EP
        dist = scipy.spatial.distance.cdist(
            [self.EP[i].objectives for i in range(l)],
            [self.EP[i].objectives for i in range(l)]
        )
        for i in range(l):
            dist[i][i] = np.inf
        dist.sort(axis=1)
        # find max self.EP_MaxSize item
        sub_dist = np.prod(dist[:, 0:self.n_objs], axis=1)
        idx = np.argpartition(sub_dist, - self.EP_MaxSize)[-self.EP_MaxSize:]
        self.EP = list((itemgetter(*idx)(self.EP)))

    def update_weight(self):
        # Delete the overcrowded subproblems
        l_ep = len(self.EP)
        nus = min(l_ep, self.nus)
        dist = scipy.spatial.distance.cdist(
            [self.population[i].objectives for i in range(
                self.population_size)],
            [self.population[i].objectives for i in range(
                self.population_size)]
        )
        for i in range(self.population_size):
            dist[i][i] = np.inf
        dist.sort(axis=1)
        sub_dist = np.prod(dist[:, 0:self.n_objs], axis=1)
        idx = np.argpartition(
            sub_dist, -(self.population_size - nus))[-(self.population_size - nus):]
        self.population = list((itemgetter(*idx)(self.population)))
        self.weights = list((itemgetter(*idx)(self.weights)))
        # Add new subproblems
        l_p = len(self.population)
        dist = scipy.spatial.distance.cdist(
            [self.EP[i].objectives for i in range(l_ep)],
            [self.population[i].objectives for i in range(l_p)]
        )  # shape = (l_ep, l_p)
        dist.sort(axis=1)
        sub_dist = np.prod(dist[:, 0:self.n_objs], axis=1)
        idx = np.argpartition(sub_dist, -nus)[-nus:]
        add_EP = list((itemgetter(*idx)(self.EP)))
        add_weights = []
        for e in add_EP:
            ans = np.asarray(e.objectives) - np.asarray(self.ideal_point)
            ans[ans < EPSILON] = 1
            ans = 1 / ans
            ans[ans == np.inf] = 1  # when f = z
            add_weights.append((ans / np.sum(ans)).tolist())
        self.population.extend(add_EP)
        self.weights.extend(add_weights)
        # Update the neighbor
        self.neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(self.population_size):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            self.neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])

    def update_igd(self, value):
        self.value_his.append(value)
        if value < self.best_value:
            self.stag_count = 0
            self.best_value = value
        else:
            self.stag_count += 1
        self.last_value = value

    def moead_update_ideal(self, solution):
        for i in range(self.problem.nobjs):
            self.ideal_point[i] = min(
                self.ideal_point[i], solution.objectives[i])

    def update_replay(self):
        self.replay_his["igd_his"].append(self.info_igd_his)
        self.replay_his["hv_his"].append(self.hv_his)
        self.replay_his["ndsort_ratio_his"].append(self.nds_ratio_his)
        self.replay_his["dis_his"].append(self.ava_dist_his)
        self.replay_his["action_his"].append(self.action_seq_his)
        self.replay_his["population_his"].append(self.population)

    def save_replay(self):
        if not os.path.exists(self.replay_dir):
            os.umask(0)
            os.makedirs(self.replay_dir, mode=0o777)
        token = f"ban{str(self.ban_agent)}_R{str(self.reward_type)}_replay.npz"
        replay_path = os.path.join(self.replay_dir, token)
        np.savez(file=replay_path, info_stack=self.replay_his)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_ref_set(self):
        return self.problem_ref_points

    def evaluate_all(self, solutions):
        unevaluated = [s for s in solutions if not s.evaluated]

        jobs = [_EvaluateJob(s) for s in unevaluated]
        results = self.evaluator.evaluate_all(jobs)

        # if needed, update the original solution with the results
        for i, result in enumerate(results):
            if unevaluated[i] != result.solution:
                unevaluated[i].variables[:] = result.solution.variables[:]
                unevaluated[i].objectives[:] = result.solution.objectives[:]
                unevaluated[i].constraints[:] = result.solution.constraints[:]
                unevaluated[i].constraint_violation = result.solution.constraint_violation
                unevaluated[i].feasible = result.solution.feasible
                unevaluated[i].evaluated = result.solution.evaluated

        self.nfe += len(unevaluated)

    def moead_calculate_fitness(self, solution, weights):
        return chebyshev(solution, self.ideal_point, weights)

    def moead_update_solution(self, solution, mating_indices):
        """
        repair solution, make constraint satisfiable
        :param solution:
        :param mating_indices:
        :return:
        """
        c = 0
        random.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            weights = self.weights[i]
            replace = False

            if solution.constraint_violation > 0.0 and candidate.constraint_violation > 0.0:
                if solution.constraint_violation < candidate.constraint_violation:
                    replace = True
            elif candidate.constraint_violation > 0.0:
                replace = True
            elif solution.constraint_violation > 0.0:
                pass
            elif self.moead_calculate_fitness(solution, weights) < self.moead_calculate_fitness(candidate, weights):
                replace = True

            if replace:
                self.population[i] = copy.deepcopy(solution)
                c = c + 1

            if c >= self.moead_eta:
                break

    @staticmethod
    def moead_sort_weights(base, weights):
        """Returns the index of weights nearest to the base weight."""

        def compare(weight1, weight2):
            dist1 = math.sqrt(
                sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(
                sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(
            enumerate(weights), key=functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def moead_get_subproblems(self):
        """
        Determines the subproblems to search.
        If :code:`utility_update` has been set, then this method follows the
        utility-based moea/D search.
        Otherwise, it follows the original moea/D specification.
        """
        indices = list(range(self.population_size))
        random.shuffle(indices)
        return indices

    def moead_get_mating_indices(self, index):
        """Determines the mating indices.

        Returns the population members that are considered during mating.  With
        probability :code:`delta`, the neighborhood is returned.  Otherwise,
        the entire population is returned.
        """
        if random.uniform(0.0, 1.0) <= self.moead_delta:
            return self.neighborhoods[index][:self.moead_neighborhood_size]
        else:
            return list(range(self.population_size))

    def get_hypervolume(self, n_samples=1e5):
        if self.problem.nobjs <= 3:
            hv_fast = False
        else:
            hv_fast = True
        if not hv_fast:
            # Calculate the exact hv value
            hyp = Hypervolume(minimum=[0 for _ in range(
                self.n_objs)], maximum=self.archive_maximum)
            hv_value = hyp.calculate(self.population)
        else:
            # Estimate the hv value by Monte Carlo
            feasible = [
                s for s in self.population if s.constraint_violation == 0.0]
            popobj = np.array(
                [feasible[i].objectives for i in range(len(feasible))])
            optimum = np.array([s.objectives for s in self.problem_ref_points])
            fmin = np.clip(np.min(popobj, axis=0), np.min(popobj), 0)
            fmax = np.max(optimum, axis=0)
            popobj = (popobj - np.tile(fmin, (self.population_size, 1))) / (
                np.tile(1.1 * (fmax - fmin), (self.population_size, 1)))
            index = np.all(popobj < 1, 1).tolist()
            popobj = popobj[index]
            if popobj.shape[0] <= 1:
                hv_value = 0
                self.hv_his.append(hv_value)
                self.hv_last5.add(hv_value)
                self.hv_running.update(np.array([hv_value]))
                return hv_value
            assert np.max(popobj) < 1
            hv_maximum = np.ones([self.n_objs])
            hv_minimum = np.min(popobj, axis=0)
            n_samples_hv = int(n_samples)
            samples = np.zeros([n_samples_hv, self.n_objs])
            for i in range(self.n_objs):
                samples[:, i] = np.random.uniform(
                    hv_minimum[i], hv_maximum[i], n_samples_hv)
            for i in range(popobj.shape[0]):
                domi = np.ones([samples.shape[0]], dtype=bool)
                m = 0
                while m < self.n_objs and any(domi):
                    domi = np.logical_and(domi, popobj[i, m] <= samples[:, m])
                    m += 1
                save_id = np.logical_not(domi)
                samples = samples[save_id, :]
            hv_value = np.prod(hv_maximum - hv_minimum) * (
                1 - samples.shape[0] / n_samples_hv)
        self.hv_his.append(hv_value)
        self.hv_last5.add(hv_value)
        self.hv_running.update(np.array([hv_value]))
        return hv_value

    def get_igd(self):
        return self.igd_calculator.calculate(self.population)

    def get_ratio_nondom_sol(self):
        self.nondominated_solution(self.population)
        count = 0
        for i in range(len(self.population)):
            if self.population[i].rank == 0:
                count = count + 1
        ratio_value = count / len(self.population)
        self.nds_ratio_his.append(ratio_value)
        self.nds_ratio_last5.add(ratio_value)
        self.nds_ratio_running.update(np.array([ratio_value]))
        return ratio_value

    def get_average_dist(self):
        total_distance = scipy.spatial.distance.cdist(
            [self.population[i].objectives for i in range(
                self.population_size)],
            [self.population[i].objectives for i in range(self.population_size)])
        ava_dist = np.mean(total_distance) / np.max(total_distance)
        self.ava_dist_his.append(ava_dist)
        self.ava_dist_last5.add(ava_dist)
        self.ava_dist_running.update(np.array([ava_dist]))
        return ava_dist

    def get_pre_k_change(self, k, value_his):
        if np.sum(self.action_count[0]) > k:
            return value_his[-1] - value_his[-(k + 1)]
        else:
            return 0

    def get_reward(self, value):
        """
        use the value to get reward
        value(default is igd), the smaller the better
        :return: reward based on current igd and historical igd
        """
        reward = 0
        if self.reward_type == 0:
            if value < self.best_value:
                bonus = (self.inital_value - value) / self.inital_value
                reward = (self.last_bonus + bonus) * (bonus - self.last_bonus)
                self.last_bonus = bonus
            reward *= 100
        elif self.reward_type == 1:
            reward = max(self.last_value - value, 0)
        elif self.reward_type == 2:
            if value < self.best_value:
                reward = 10
            elif value < self.last_value:
                reward = 1
        elif self.reward_type == 3:
            reward = max((self.last_value - value) / value, 0)
        else:
            raise ValueError("Invaild Reward Type.")
        return reward
