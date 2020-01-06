import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random, os
import json

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Linear(10,1024)
        # self.bn1 = nn.BatchNorm1d(1568)
        self.conv2 = nn.Linear(1024,128)
        # self.bn2 = nn.BatchNorm1d(100)
        self.conv3 = nn.Linear(128,32)
        # self.bn3 = nn.BatchNorm1d(30)
        self.head  = nn.Linear(32,outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x.float())))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x))) 
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        return self.head(x.view(x.size(0), -1))

    
def evaluate_episode (episode_record):
    reward_list = [sum(r[1]) for r in episode_record]
    if len(reward_list) == 0:
        print ('error: reward list is 0 length, something must be wrong!')
        return 0
    return sum(reward_list)/len(reward_list)

def get_first_usable_name (model_name):
    # increase index until the first unused name
    i = 1
    while os.path.exists('params/{}_{}'.format(model_name, i)):
        i += 1
    return 'params/{}_{}'.format(model_name, i)

def create_saving_folder (model_name, cmd_args):
    # this function will create the folder of interesting
    # return the name of folder created
    folder = get_first_usable_name( model_name )
    os.mkdir(folder)
    with open(folder+'/args.txt', 'w') as f:
        json.dump(cmd_args.__dict__, f, indent=2)
    return folder


if __name__ == '__main__':
    print ('this is util file, do not run directly')
