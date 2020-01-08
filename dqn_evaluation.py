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


import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper
args = TrafficEnv.get_default_init_parameters()
def evaluate(network, env, device=None, total_step = 2999):
  if not device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # return the waiting time of a policy network on a certain env
  next_state = env.reset()
  for _ in range(0, total_step):
    actions = network(torch.tensor(next_state).to(device))
    action = actions.max(1)[1].view(1, 1)
    next_state, reward, terminal, _ = env.step([action.item()])
  return env.get_waiting_time()

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
  parser.add_argument('--visual', action='store_true', help='use visualization')
  parser.add_argument('--filename', action='store', type=str, help='path of the model parameter')
  cmd_args = parser.parse_args()
  #print(args)
  env = gym.make('TrafficLight-v0')
  
  if cmd_args.visual:
    args['visual'] = True
  env = TrafficParameterSetWrapper(env, args)
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")
  screen_height, screen_width = env.observation_space.shape
  n_actions = env.action_space.n
  target_net = DQN(screen_height, screen_width, n_actions).to(device)
  target_net.load_state_dict(torch.load(cmd_args.filename))

  itr = 0
  next_state = env.reset()
  while itr < 3000:
    itr += 1
    actions = target_net(torch.tensor(next_state))
    #print (actions)
    action = actions.max(1)[1].view(1, 1)
    # print(action)
    next_state, reward, terminal, _ = env.step([action.item()])
    # print (next_state)