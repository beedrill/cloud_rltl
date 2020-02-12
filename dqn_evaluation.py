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
import sys

import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper

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
  parser.add_argument('--delay', action='store', type=int, help='network delay')
  parser.add_argument('--env_name', action='store', type=str, help='environment name')
  parser.add_argument('--saving_file', action='store', type=str, help='saving file')
  parser.add_argument('--n_trials', action='store', default = 1, type=int, help='number of trials to evaluate')
  cmd_args = parser.parse_args()
  #print(args)
  if not cmd_args.env_name:
    print ('need env name, use --env_name to specify')
    sys.exit(1)

  if not cmd_args.saving_file:
    print ('need to know saving file, use --saving_file')
    sys.exit(1)
  env = gym.make(cmd_args.env_name)
  args = {}
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if cmd_args.visual:
    args['visual'] = True
    device = torch.device('cpu') #always use cpu when need to visualize (*Is there a fix for this? Is it necessary to fix this?*)
  if cmd_args.delay:
    args['action_delay'] = cmd_args.delay
  env = TrafficParameterSetWrapper(env, args)
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = env.unwrapped
  
  screen_height, screen_width = env.observation_space.shape
  n_actions = env.action_space.n
  target_net = DQN(screen_height, screen_width, n_actions).to(device)
  target_net.load_state_dict(torch.load(cmd_args.filename))
  t_waiting = 0
  for _ in range(0, cmd_args.n_trials):
    waiting_times = evaluate(target_net, env)
    t_waiting += waiting_times[0]
  avg_waiting_time = t_waiting/float(cmd_args.n_trials)
  with open(cmd_args.saving_file, 'a') as f:
    f.write('for env {}, delay {}, waiting time is: {}\n'.format(cmd_args.env_name, cmd_args.delay, avg_waiting_time))