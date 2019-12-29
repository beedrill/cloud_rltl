import gym
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import DQN, ReplayMemory
import argparse
import torch
parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
parser.add_argument('--visual', action='store_true', help='use visualization')
cmd_args = parser.parse_args()

import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper
args = TrafficEnv.get_default_init_parameters()
if cmd_args.visual:
  args['visual'] = True
print(args)
env = gym.make('TrafficLight-v0')
env = TrafficParameterSetWrapper(env, args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
screen_height, screen_width = env.observation_space.shape
n_actions = env.action_space.n
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(torch.load('params/net_params_100.pkl'))

itr = 0
state = env.reset()
while itr < 3000:
  itr += 1
  action = target_net(torch.tensor(state)).max(1)[1].view(1, 1)
  # print(action)
  next_state, reward, terminal, _ = env.step([action.item()])
  # print (next_state)