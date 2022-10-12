import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(
    dirname(dirname(dirname(abspath(__file__)))), "results", "madac")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

# get config from argv, such as "remark"


def _get_argv_config(params):
    config = {}
    to_del = []
    for _i, _v in enumerate(params):
        item = _v.split("=")[0]
        if item[:2] == "--" and item not in ["envs", "algs"]:
            config_v = _v.split("=")[1]
            try:
                config_v = eval(config_v)
            except:
                pass
            config[item[2:]] = config_v
            to_del.append(_v)
    for _v in to_del:
        params.remove(_v)
    return config


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    config_dict = recursive_dict_update(config_dict, _get_argv_config(params))

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}")

    config_dict["env_args"]["replay_dir"] = os.path.join(
        results_path, f"{alg_config['name']}_{map_name.split('_')[-1]}_{config_dict['load_step']}", map_name)

    # now add all the config to sacred
    ex.add_config(config_dict)

    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
