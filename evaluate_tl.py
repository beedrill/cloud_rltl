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
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
  parser.add_argument('--env_name', action='store', type=str, help='environment name')

  cmd_args = parser.parse_args()
  #print(args)
  if not cmd_args.env_name:
    print ('need env name, use --env_name to specify')
    sys.exit(1)
  env = gym.make(cmd_args.env_name)
  env.reset()
  for _ in range(0, 2999):
    t = 0
    if t == 30:
        t = 0
        env.step([1])
    else:
        env.step([0])
    t+=1
    
  print (env.get_waiting_time())