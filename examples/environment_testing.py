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
print(args)
env = gym.make('TrafficLight-v0')
env = TrafficParameterSetWrapper(env, args)
#args={'penetration_rate': 0.5}
itr = 0
env.reset()
while itr < 3000:
  itr += 1
  next_state, reward, terminal, _ = env.step([0])
  # print (next_state)

env.reset()