import copy
import numpy as np


def neighbor_size(action):
    # Agent 0
    ns = [15, 20, 25, 30]
    return ns[action]


def operator_parameter(action):
    # Agent 2
    scale = [0.4, 0.5, 0.6, 0.7]
    return scale[action]


def de_item(*args):
    res = None
    for idx, item in enumerate(args):
        if idx == 0:
            res = np.asarray(item)
        else:
            f = (1 if (idx % 2 == 0) else -1)
            res += (f * np.asarray(item))
    return res


def operator_type(action, population, xi, de_pool, scale, lb, ub):
    """
    Agent 1
    @param action:
    @param population:
    @param xi: current subproblem individual
    @param de_pool:
    @param scale: Hyperparameters, In MOEA/D FRRMAB, default F=K=0.5, CR=1.0
    @param lb:
    @param ub:
    @return:
    """
    x_r = []
    for i in range(5):
        x_r.append(population[de_pool[i]])
    x_i = population[xi]
    child = copy.deepcopy(x_i)
    child_var = np.asarray(child.variables)
    if action == 0:  # DE/rand/1
        child_var += scale * de_item(x_r[0].variables, x_r[1].variables)
    elif action == 1:  # DE/rand/2
        child_var += scale * \
            de_item(x_r[0].variables, x_r[1].variables,
                    x_r[2].variables, x_r[3].variables)
    elif action == 2:  # DE/rand-to-best/2
        child_var += scale * \
            de_item(x_i.variables, x_r[0].variables, x_r[1].variables,
                    x_r[2].variables, x_r[3].variables, x_r[4].variables)
    elif action == 3:  # DE/current-to-rand/1
        child_var += scale * \
            de_item(x_i.variables, x_r[0].variables,
                    x_r[1].variables, x_r[2].variables)
    else:
        raise ValueError("Invalid operator type action.")
    child.variables = np.clip(child_var, lb, ub).tolist()
    child.evaluated = False
    return child


def test_de_item():
    a = [1, 2, 3]
    b = [30, 20, 10]
    c = [100, 200, 300]
    print(de_item(a, b, c))


if __name__ == "__main__":
    test_de_item()
