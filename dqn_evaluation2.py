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

#args = TrafficEnv.get_default_init_parameters()
args = {}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
    parser.add_argument('--visual', action='store_true', help='use visualization')
    parser.add_argument('--folder_name', action='store', type=str, help='path of the model parameter')
    parser.add_argument('--delay_time', action = 'store', default = 0, type = int, help = 'specify the delay time')
    parser.add_argument('--env_option', action='store', default=0, type=int, help='specify the environment')
    cmd_args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(args)
    # env = gym.make('TrafficLight-v0')
    #env = gym.make('TrafficLight-simple-sparse-v0')
    #env = gym.make('TrafficLight-simple-medium-v0')
    #env = gym.make('TrafficLight-simple-dense-v0')
    #env = gym.make('TrafficLight-Lust12408-rush-hour-v0')
    #env = gym.make('TrafficLight-Lust12408-regular-time-v0')
    #env = gym.make('TrafficLight-Lust12408-midnight-v0')
    if cmd_args.env_option == 0:
        env = gym.make('TrafficLight-v0')
        env_name = "v0"
    elif cmd_args.env_option == 1:
        env = gym.make('TrafficLight-simple-sparse-v0')
        env_name = "simple_sparse"
    elif cmd_args.env_option == 2:
        env = gym.make('TrafficLight-simple-medium-v0')
        env_name = "simple_medium"
    elif cmd_args.env_option == 3:
        env = gym.make('TrafficLight-simple-dense-v0')
        env_name = "simple_dense"
    elif cmd_args.env_option == 4:
        env = gym.make('TrafficLight-Lust12408-rush-hour-v0')
        env_name = "Lust12408_rush"
    elif cmd_args.env_option == 5:
        env = gym.make('TrafficLight-Lust12408-regular-time-v0')
        env_name = "Lust12408_regular"
    elif cmd_args.env_option == 6:
        env = gym.make('TrafficLight-Lust12408-midnight-v0')
        env_name = "Lust12408_midnight"
    if cmd_args.visual:
        args['visual'] = True
    env = TrafficParameterSetWrapper(env, args)
    env = env.unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    screen_height, screen_width = env.observation_space.shape
    n_actions = env.action_space.n
    wait_list = []
    folder_name = 'DRL' + '_' + env_name + '_' + 'delay_time' + '_' + str(cmd_args.delay_time) + '_1'
    for i in range(150):
        filename  = "./params/%s/net_params_%d.pkl" % (folder_name, i)
        target_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net.load_state_dict(torch.load(filename))
        env.reset()
        waiting_times = evaluate(target_net, env, device=device)
        wait_list.append((i,waiting_times[0]))
    wait_list.sort(key = lambda k:k[1])
    best_network_index = wait_list[0][0]
    # Re-run 5 times, take average
    filename = "./params/%s/net_params_%d.pkl" % (folder_name, best_network_index)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(torch.load(filename))
    wait_total = 0
    for i in range(5):
        env.reset()
        waiting_times = evaluate(target_net, env, device=device)
        wait_total += waiting_times[0]
    wait_average = wait_total / 5.0

    print('env_%s delay_time %d average waiting time %f\n' % (env_name, cmd_args.delay_time, wait_average))