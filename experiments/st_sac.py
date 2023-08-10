from metaworld_utils import generate_single_mt_env
# evaluation on single tasks using double q sac

import sys
sys.path.append(".")

import torch

import os
import time
import os.path as osp
import copy
import numpy as np
import utils

args = utils.get_args()
params = utils.get_params(args.config)


from metaworld_utils.meta_env import generate_single_mt_env
import gym
from metaworld_utils.meta_env import get_meta_env
from spinningup.spinup.utils.run_utils import ExperimentGrid
from spinningup.spinup import sac_pytorch
import torch
import random

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    print("using device: ", device.type)
    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True

    tasks = list(cls_dicts.keys())
    task_name = tasks[args.task_id]


    single_env_args = {
                "task_cls": cls_dicts[task_name],
                "task_args": copy.deepcopy(cls_args[task_name]),
                "env_rank": args.task_id,
                "num_tasks": len(tasks),
                "max_obs_dim": np.prod(env.observation_space.shape),
                "env_params": params["env"],
                "meta_env_params": params["meta_env"],
            }
    if "start_epoch" in single_env_args["task_args"]:
        del single_env_args["task_args"]["start_epoch"]

    
    env_fn = lambda: generate_single_mt_env(**single_env_args)
    eg = ExperimentGrid(name=args.id)
    eg.add('env_fn', env_fn)
    for k, v in params['sac']:
        eg.add(k, v)
    eg.add('ac_kwargs:hidden_sizes', [tuple(params['ac_kwargs']['hidden_sizes']), tuple(params['ac_kwargs']['qf_hidden_sizes'])])
    eg.add('ac_kwargs:activation', [torch.nn.ReLU, torch.nn.ReLU]) 
    logger_kwargs = dict(output_dir='log/MT1_single_tasks', exp_name=args.id)
    eg.add('logger_kwargs', logger_kwargs)
    eg.run(sac_pytorch, num_cpu=4)

    print("learning on task: ", task_name)
    
    

tasks = [
    'reach-v1', 
    'push-v1', 
    'pick-place-v1', 
    'door-v1', 
    'drawer-open-v1', 
    'drawer-close-v1', 
    'button-press-topdown-v1', 
    'ped-insert-side-v1', 
    'window-open-v1', 
    'window-close-v1'
]

cfg_idxs = [2, 3, 4, 5, 6]
#if __name__ == "__main__":
#    for cfg in cfg_idxs:
#        args.config = f"meta_config/mt1/{tasks[args.task_id]}_sac copy {cfg}.json"
#        args.seed = cfg+10
#        params = get_params(args.config)
#        experiment(args)

if __name__ == "__main__":
    experiment(args)
'''
 CODE NAMES
0 'reach-v1', 
1 'push-v1', 
2 'pick-place-v1', 
3 'door-v1', 
4 'drawer-open-v1', 
5 'drawer-close-v1', 
6 'button-press-topdown-v1', 
7 'ped-insert-side-v1', 
8 'window-open-v1', 
9 'window-close-v1'

'''