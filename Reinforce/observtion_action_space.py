#!/usr/bin/env python3


import gym
from gym.spaces import *
import sys


def print_spaces(space):
    
    print(space)
    
    if isinstance(space,Box):
        print('\n space.low, ',space.low)
        print('\n space.high, ',space.high)
        

if __name__ == "__main__":
    
    env = gym.make("MountainCar-v0")
    print("observation space")
    print_spaces(env.observation_space)
    print('Action space')
    print_spaces(env.action_space)
    
    try:
        print("Action description/meaning:",env.unwrapped.get_action_meanings())
    except AttributeError:
        pass



