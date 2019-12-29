import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        self.conv1 = nn.Linear(10,1568)
        # self.bn1 = nn.BatchNorm1d(1568)
        self.conv2 = nn.Linear(1568,100)
        # self.bn2 = nn.BatchNorm1d(100)
        self.conv3 = nn.Linear(100,30)
        # self.bn3 = nn.BatchNorm1d(30)
        self.head  = nn.Linear(30,outputs)

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