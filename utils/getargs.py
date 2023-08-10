import argparse
import json

import torch

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
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')

    parser.add_argument("--config", type=str,   default=None,
                        help="config file", )

    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')
    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )

    # single env
    parser.add_argument("--task_id", type=int, default=-1,
                        help="id(int) of task to train on")

    args = parser.parse_args()
    if args.task_id >= 0: 
        args.config = f"config/{tasks[args.task_id]}_sac.json"

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params
