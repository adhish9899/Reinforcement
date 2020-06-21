
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
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START
    
    return next_state, reward
    
# reward for each action in each state
# action_rewards = np.zeros(WORLD_HEIGHT, WORLD_WIDTH, 4)
# action_rewards[:,:,:] = -1.0
# action_rewards[2, 1:11, ACTION_DOWN] = -100
# action_rewards[3, 0 , ACTION_RIGHT] = -100

# Choose an action based on epsilon greedy policy
def choose_action(state, q_value):

    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_  in enumerate(values_) if values_ == np.max(values_)])

# an episode with SARSA
# @q_value: values for state, action pair will be upgraded
# @expected: if True, it will use expected SARSA algorithm
# @step_size: step size for updateing
# @return: total rewards within this episode

def sarsa(q_value, expected=False, step_size=ALPHA):

    state = START
    action = choose_action(state, q_value)
    rewards = 0.0

    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward

        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        
        else:
            # Calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_action = np.argwhere(q_next == np.max(q_next))

            for action_ in ACTIONS:
                if action_ in best_action:
                    target += ((1 - EPSILON)/len(best_action) + EPSILON/len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]

                else:
                    target += (EPSILON/len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
            
        target *= GAMMA

        #Updating current state action values
        q_value[state[0], state[1], action] += step_size( reward + target - q_value[state[0], state[1], action])

        state = next_state
        action = next_action
    
    return rewards

# an episode with Q-Learning
# @q_value: values for state, action will be updated
# @step_size: step size for updating
# @return: total rewards within this episode

def q_learning(q_value, step_size=ALPHA):

    state = START
    rewards = 0.0

    while state != GOAL:

        action = choose_action(state, q_value)  
        next_state, reward = step(state, action)

        rewards += reward

        # Q Learning update
        q_value[state[0], state[1], action] += step_size * (reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                                                            q_value[state[0], state[1], action])                 

        state = next_state

    return rewards

# Print optimal policy
def print_optimal_policy(q_value):

    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):

            if [i, j] == GOAL:
                optimal_policy[-1].append("G")
                continue

            best_action = np.argmax(q_value[i, j, :])
            if best_action == ACTION_UP:
                optimal_policy[-1].append("U")
            
            elif best_action == ACTION_LEFT:
                optimal_policy[-1].append("L")
            
            elif best_action == ACTION_DOWN:
                optimal_policy[-1].append("D")
            
            elif best_action == ACTION_RIGHT:
                optimal_policy[-1].append("R")
    
    for row in optimal_policy:
        print(row)


# Use multiple runs 





