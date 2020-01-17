import gym
import math
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_evaluation import evaluate
from utils import DQN, ReplayMemory, Transition, evaluate_episode, create_saving_folder

#################parsing arguments#####################################
import argparse
parser = argparse.ArgumentParser(description='Run Q learning for cloud rltl')
parser.add_argument('--visual', action='store_true', help='use visualization')
parser.add_argument('--no_normalize_reward', action='store_true', help='do not normalize reward')
parser.add_argument('--lr', action='store', default=0.0001, type=float, help='specify learning rate, default is 0.001')
parser.add_argument('--epsilon_start', action='store', default=0.5, type=float, help='exploration rate at beginning of the training, default is 0.1')
parser.add_argument('--epsilon_end', action='store', default=0.001, type=float, help='exploration rate at end of the training, default is 0.001')
parser.add_argument('--epsilon_decay', action='store', default=100000, type=int, help='number of steps that epsilon reaches epsilon_end, default is 100,000')
parser.add_argument('--target_update', action='store', default=3000, type=int, help='number of steps for targets to update, default is 2')
parser.add_argument('--gamma', action='store', default=0.9, type=float, help='reward decay factor, default is 0.99')
parser.add_argument('--model_name', action='store', default='DRL', type=str, help='the name of the run, default is DQN')
parser.add_argument('--replay_memory_size', action='store', default=200000, type=int, help='memory replay buffer size')
parser.add_argument('--batch_size', action='store', default=32, type=int, help='mini batch size')
parser.add_argument('--evaluation_gap', action = 'store', default = 3, type = int, help = 'how many episode to evaluate')
parser.add_argument('--delay_time', action = 'store', default = 0, type = int, help = 'specify the delay time')
parser.add_argument('--env_option', action = 'store', default = 0, type = int, help = 'specify the environment')
cmd_args = parser.parse_args()
#######################################################################

import gym
import gym_trafficlight
from gym_trafficlight.trafficenvs import TrafficEnv
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper
args = {}
if cmd_args.visual:
  args['visual'] = True
args['reward_present_form'] = 'reward' # we use reward as opposed to penalty
if cmd_args.no_normalize_reward:
  args['normalize_reward'] = False

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
elif cmd_args.env_option == 7:
    env = gym.make('CartPole-v0')
    env_name = "Lust12408_midnight"
env = TrafficParameterSetWrapper(env, args)
env = env.unwrapped

# env = gym.make('CartPole-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('checking device... the computation device used in the training is: ' + str(device))

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

saving_name   = cmd_args.model_name + '_' + env_name + '_' + cmd_args.delay_time
saving_folder = create_saving_folder(saving_name, cmd_args)
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

BATCH_SIZE = cmd_args.batch_size
GAMMA = cmd_args.gamma
EPS_START = cmd_args.epsilon_start
EPS_END = cmd_args.epsilon_end
EPS_DECAY = cmd_args.epsilon_decay
TARGET_UPDATE = cmd_args.target_update
EVALUATION_GAP = cmd_args.evaluation_gap
DELAY_TIME = cmd_args.delay_time

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

screen_height, screen_width = env.observation_space.shape
# for traffic env, the height and width are not really screen size, but we use the traditional denotation
print('the size of env is: ' + str(screen_width) + ', ' + str(screen_height))
# Get number of actions from gym action space
n_actions = env.action_space.n
print('the size of action is: ' + str(n_actions))
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # copy the policy net param to target net
target_net.eval() #set target network in evaluation mode

#optimizer = optim.RMSprop(policy_net.parameters(), lr = cmd_args.lr)
optimizer = optim.Adam(policy_net.parameters(), lr = cmd_args.lr)
memory = ReplayMemory(cmd_args.replay_memory_size)


steps_done = 0


def select_action(state, buffered_action):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    if len(buffered_action) == 0:
        delay_flag = False
    else:
        delay_flag = True

    if delay_flag:
        flag = False
        # Calculate buffered action
        if buffered_action[0] == -1:
            flag = True
            # print("Initialization exception")
            for i in range(DELAY_TIME):
                # actions = policy_net(state)
                actions = policy_net(torch.tensor(state).to(device))
                action = actions.max(1)[1].view(1, 1)
                state, reward, terminal, _ = env.step(action)
                if i == 0:
                    first_action = action
                else:
                    buffered_action[i-1] = action
        else:
            first_action = buffered_action[0]
            for i in range(DELAY_TIME):
                state, reward, terminal, _ = env.step([buffered_action[i]])
        # print (actions)
        actions = policy_net(torch.tensor(state).to(device))
        action = actions.max(1)[1].view(1, 1)
        if flag:
            buffered_action[-1] = action
            # print("buffered action")
            # print(buffered_action)
        else:
            buffered_action = torch.cat((torch.tensor(buffered_action[1:],device=device), torch.tensor([action],device=device)), 0).to(device)
    else:
        actions = policy_net(torch.tensor(state).to(device))
        first_action = actions.max(1)[1].view(1, 1)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.tensor(first_action), buffered_action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), buffered_action




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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8).to(device)
    
    #print("non_final_mask:",non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    #print("non_final_next_states", non_final_next_states)
    #print("non_final_next_states shape", non_final_next_states.shape)
    # (batch_size, state_h, state_w)
    state_batch  = torch.cat(batch.state).to(device)
    # print(batch.action)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    #print("reward batch:", reward_batch.shape)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("state batch shape:",state_batch.shape)
    # print("action_batch:", action_batch)
    # print("unsqueeze:",action_batch.unsqueeze(1).shape)
    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch.unsqueeze(1).view(BATCH_SIZE,1))
    #print("state action values:",state_action_values)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #print("next_state_values:", next_state_values.shape)
    # Compute the expected Q values
    expected_state_action_values = ((next_state_values.view(BATCH_SIZE,1).type(torch.DoubleTensor).to(device) * GAMMA) + reward_batch)
    #print("state action values size:",state_action_values.shape)
    #print("expected state action values:",expected_state_action_values)
    #print("expected state action values size:", expected_state_action_values.unsqueeze(1)[:,:,0].shape)
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.view(BATCH_SIZE,1), expected_state_action_values.unsqueeze(1).view(BATCH_SIZE,1).float()).to(device)
    #print("loss",loss)
    #input('press to continue')
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
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
#print("Initial State")
#print(state)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    buffered_action = torch.tensor([-1] * DELAY_TIME).to(device)
    episode_record = [] # use this to record temporarily for one episode
    # for t in count():
    for t in range(2999):
        steps_done += 1
        # Select and perform an action
        # print(state.shape)
        action, buffered_action = select_action(torch.tensor(state).to(device), buffered_action)
        # print(buffered_action)
        next_state, reward, terminal, _ = env.step([action])
        episode_record.append((next_state, reward))
        # print(next_state.shape)
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory
        # print("push action %s"%buffered_action[0])
        # print(cur_action)
        memory.push(torch.tensor([state]), torch.tensor(action.view(1,1)), torch.tensor([next_state]), reward)
        # print("reward",reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        # if terminal:
        #     print('terminal',t)
        #     episode_durations.append(t + 1)
        #     break
        # Update the target network, copying all weights and biases in DQN
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    average_reward = evaluate_episode(episode_record)
    print("episode:", i_episode, 'average reward:', average_reward)
    torch.save(target_net.state_dict(),saving_folder+'/net_params_%d.pkl'%(i_episode)) 
    if i_episode%EVALUATION_GAP == 0:
        waiting_times = evaluate(target_net, env, device = device)
        print ('average waiting time: ', waiting_times[0])

print('Complete')

# env.render()
env.close()
