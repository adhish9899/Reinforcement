
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# Probability of exploration (EPSILON)
EPSILON = 0.1

# Step Size (alpha)
ALPHA = 0.5

# Gamma for Q-learning and expected SARSA
GAMMA = 0.1

# all possible ACTIONS
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [3,0]
GOAL = [3,11]

def step(state, action):
    i,j = state
    if action == ACTION_UP:
        next_state = [max(0, i-1), j]
    
    elif action == ACTION_DOWN:
        next_state = [min(WORLD_HEIGHT - 1, i+1), j]
    
    elif action == ACTION_LEFT:
        next_state = [i, max(0, j-1)]
    
    elif action == ACTION_RIGHT:
        next_state = [i, min(WORLD_WIDTH - 1, j+1)]
    
    else:
        raise IndexError("Your action {} is not valid".format(action))
    
    reward = -1
    if (action == ACTION_DOWN and i == 2 and 2 <= j <= 10) or (action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START
    
    return next_state, reward
    




