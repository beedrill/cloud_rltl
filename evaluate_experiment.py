import gym
import math
import random
import numpy as np
from collections import namedtuple
import sys, os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_evaluation import evaluate
from utils import DQN, ReplayMemory
import argparse
import torch
import json
import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import TrafficParameterSetWrapper

#args = TrafficEnv.get_default_init_parameters()
def find_trial_directory(path):
    #this function will return all applicable folders to do evaluations one by one
    trial_dirs = []
    for x in os.listdir(path):
        if os.path.isdir(os.path.join(path, x)):
            p = os.path.join(path, x, 'args.txt') # if args.txt file exists, we consider the folder contain the params run by the training file
            if os.path.isfile(p):
                trial_dirs.append(os.path.join(path, x))
    return trial_dirs

def build_env_args(exp_args):
    args = {}
    args['action_delay'] = exp_args['delay']
    return args

def evaluate_trial(args, n_trials):
    # env = gym.make(args['env_name'])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env_args = build_env_args(args)
    # env = TrafficParameterSetWrapper(env, env_args)
    # env = env.unwrapped
    # screen_height, screen_width = env.observation_space.shape
    # n_actions = env.action_space.n
    # target_net = DQN(screen_height, screen_width, n_actions).to(device)
    # target_net.load_state_dict(torch.load(os.path.join('params', args['model_saving_path'], 'best_net_params.pkl')))
    #wait_total = 0
    #for _ in range(n_trials):
    model_path = os.path.join('params', args['model_saving_path'], 'best_net_params.pkl')
    saving_path = os.path.join('params', args['model_saving_path'], os.pardir, 'result.txt')
    os.system('python3 dqn_evaluation.py --filename {} --delay {} --env_name {} --saving_file {} --n_trials {}'.format(model_path, 
    args['delay'], args['env_name'], saving_path, n_trials))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of Q learning for cloud rltl experiments')
    parser.add_argument('--visual', action='store_true', help='use visualization')
    # parser.add_argument('--folder_name', action='store', type=str, help='path of the model parameter')
    # parser.add_argument('--delay_time', action = 'store', default = 0, type = int, help = 'specify the delay time')
    # parser.add_argument('--env_option', action='store', default=0, type=int, help='specify the environment')
    parser.add_argument('-p', '--experiment_path',  action='store', default='', type=str, help='experiment folder')
    cmd_args = parser.parse_args()
    if not cmd_args.experiment_path:
        print('use -p to specify the experiment folder you used for training')
        sys.exit(1)
    else:
        print ('the path to experiment is: {}'.format(cmd_args.experiment_path))
    
    dirs = find_trial_directory(cmd_args.experiment_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dir in dirs:
        exp_args = json.load(open(os.path.join(dir, 'args.txt'))) #load experimental setting parameters
        evaluate_trial(exp_args, 5)
        #print ('env {} delay {} result: {}'.format(exp_args['env_name'], exp_args['delay'], t))

    # env = TrafficParameterSetWrapper(env, args)
    # env = env.unwrapped
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # screen_height, screen_width = env.observation_space.shape
    # n_actions = env.action_space.n
    # wait_list = []
    # folder_name = 'DRL' + '_' + env_name + '_' + 'delay_time' + '_' + str(cmd_args.delay_time) + '_1'
    # for i in range(150):
    #     filename  = "./params/%s/net_params_%d.pkl" % (folder_name, i)
    #     target_net = DQN(screen_height, screen_width, n_actions).to(device)
    #     target_net.load_state_dict(torch.load(filename))
    #     env.reset()
    #     waiting_times = evaluate(target_net, env, device=device)
    #     wait_list.append((i,waiting_times[0]))
    # wait_list.sort(key = lambda k:k[1])
    # best_network_index = wait_list[0][0]
    # # Re-run 5 times, take average
    # filename = "./params/%s/net_params_%d.pkl" % (folder_name, best_network_index)
    # target_net = DQN(screen_height, screen_width, n_actions).to(device)
    # target_net.load_state_dict(torch.load(filename))
    # wait_total = 0
    # for i in range(5):
    #     env.reset()
    #     waiting_times = evaluate(target_net, env, device=device)
    #     wait_total += waiting_times[0]
    # wait_average = wait_total / 5.0

    # print('env_%s delay_time %d average waiting time %f\n' % (env_name, cmd_args.delay_time, wait_average))