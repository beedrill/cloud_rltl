import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import DQN, ReplayMemory

import argparse
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
env = gym.make('TrafficLight-v0')
env = TrafficParameterSetWrapper(env, args)


# env = gym.make('CartPole-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env.reset()

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.001
EPS_DECAY = 50
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

screen_height, screen_width = env.observation_space.shape
# print(screen_width,screen_height)

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # print("trasitions:", transitions)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # Batch is a name tuple, each field contains a list of batch size states.
    batch = Transition(*zip(*transitions))
    # print("batch", batch)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    # print("non_final_mask:",non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # print("non_final_next_states", non_final_next_states)
    # print("non_final_next_states shape", non_final_next_states.shape)
    # (batch_size, state_h, state_w)
    state_batch  = torch.cat(batch.state)
    # print(state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print("reward batch:", reward_batch.shape)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("state batch shape:",state_batch.shape)
    # print("action_batch:", action_batch)
    # print("unsqueeze:",action_batch.unsqueeze(1))
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    # print("state action values:",state_action_values)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # print("next_state_values:", next_state_values.shape)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.view(BATCH_SIZE,1) * GAMMA) + reward_batch
    # print("state action values size:",state_action_values.shape)
    # print("expected state action values:",expected_state_action_values)
    # print("expected state action values size:", expected_state_action_values.unsqueeze(1)[:,:,0].shape)
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.view(BATCH_SIZE,1), expected_state_action_values.unsqueeze(1).view(BATCH_SIZE,1).float())
    # print("loss",loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#

num_episodes = 150
state = env.reset()
print("Initial State")
print(state)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    print("episode:", i_episode)
    # for t in count():
    for t in range(1000):
        # Select and perform an action
        # print(state.shape)
        action = select_action(torch.tensor(state))
        # print(action.item())
        next_state, reward, terminal, _ = env.step([action.item()])
        # print(next_state.shape)
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory
        memory.push(torch.tensor([state]), torch.tensor([action]), torch.tensor([next_state]), reward)
        # print("reward",reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if terminal:
            print('terminal')
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    torch.save(target_net.state_dict(),'params/net_params_%d.pkl'%(i_episode))

print('Complete')

# env.render()
env.close()