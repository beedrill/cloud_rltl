import gym
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_evaluation import evaluate
from utils import DQN, ReplayMemory
import argparse
import torch

import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import TrafficParameterSetWrapper

args = TrafficEnv.get_default_init_parameters()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
    parser.add_argument('--visual', action='store_true', help='use visualization')
    parser.add_argument('--folder_name', action='store', type=str, help='path of the model parameter')
    cmd_args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(args)
    env = gym.make('TrafficLight-v0')

    if cmd_args.visual:
        args['visual'] = True
    env = TrafficParameterSetWrapper(env, args)
    env = env.unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    screen_height, screen_width = env.observation_space.shape
    n_actions = env.action_space.n
    wait_list = []
    for i in range(12):
        filename  = "./params/DRL_%s/net_params_%d.pkl"%(cmd_args.folder_name, i)
        target_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net.load_state_dict(torch.load(filename))
        env.reset()
        waiting_times = evaluate(target_net, env, device=device)
        wait_list.append((i,waiting_times[0]))
    wait_list.sort(key = lambda k:k[1])
    best_network_index = wait_list[0][0]
    # Re-run 5 times, take average
    filename = "./params/DRL_%s/net_params_%d.pkl" % (cmd_args.folder_name, best_network_index)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(torch.load(filename))
    wait_total = 0
    for i in range(5):
        env.reset()
        waiting_times = evaluate(target_net, env, device=device)
        wait_total += waiting_times[0]
    wait_average = wait_total / 5.0

    with open('./evaluation/DRL_%s_net_params_%d.txt' % (cmd_args.folder_name, best_network_index),'w+') as f:
        f.write("DRL_%s best network average waiting time %f\n" % (cmd_args.folder_name, wait_average))

    print('DRL_%s best network average waiting time %f\n' % (cmd_args.folder_name, wait_average))