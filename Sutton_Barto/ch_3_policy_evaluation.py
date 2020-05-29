
### GRID WORLD EXAMPLE MDP (SUTTON AND BARTO)

import numpy as np
import copy
from math import *
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns

def action_reward(intial_position, action, terminal_states, reward_size):

    if intial_position in terminal_states:
        return intial_position, 0
    
    final_position = np.array(intial_position) + np.array(action)

    if -1 in final_position or 4 in final_position:
        final_position = intial_position
    
    return final_position, reward_size


def gird_world():
    gamma = 1
    reward_size = -1
    grid_size = 4
    termination_states = [[0,0], [grid_size-1, grid_size-1]]
    actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    num_iterations = 1000

    value_map = np.zeros((grid_size, grid_size))
    states = [[i, j] for i in range(grid_size) for j in range(grid_size)]

    deltas = []
    for it in range(num_iterations):
        copy_value_map = np.copy(value_map)
        delta_state =  []
        for state in states:
            weighted_rewards = 0
            for action in actions:
                final_position, reward = action_reward(state, action, termination_states, reward_size)

                weighted_rewards += (1/len(actions))*(reward + gamma*value_map[final_position[0], final_position[1]])

            delta_state.append(np.abs(copy_value_map[state[0], state[1]] - weighted_rewards))
            copy_value_map[state[0], state[1]] = weighted_rewards
        
        deltas.append(delta_state)
        value_map = copy.deepcopy(copy_value_map)
        if it in [0,1,2,9, 99, num_iterations-1]:
            print("Iteration {}".format(it+1))
            print(value_map)
            print("")


def Actions(s, goal):
    possible_stakes = np.arange(0,min(s, goal - s)+1)
    return possible_stakes

def gamblers_problem():

    goal = 100
    states = np.arange(goal + 1)
    head_prob = 0.4

    state_value = np.zeros(goal + 1)
    state_value[goal] = 1.0

    sweeps_history = []

    # Value Iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in states[1:goal]:
            # Get all possible actions
            actions = np.arange(min(state, goal-state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action])
            
            new_value = np.max(action_returns)
            state_value[state] = new_value
        
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    policy = np.zeros(goal + 1)
    for state in states[1:goal]:
        actions = np.arange(min(state, goal-state) + 1)
        action_returns = []

        for action in actions:
            action_returns.append(head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action])
        
        # Taking from 1 as the first action is placing "0" bet.
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(states, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.show()


# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()

def jacks_car_rental():

    max_cars = 20
    max_move_cars = 5

    # expectation for rental requests in first location
    rental_request_first_loc = 3

    # expectation for rental requests in second location
    rental_request_second_loc = 4

    # Expected no. of cars returned in the first location
    returns_first_loc = 3

    # Expected no. of cars returned in the second location
    returns_second_loc = 2

    gamma = 0.9

    # credit earned by a car
    rental_credits = 10

    # cost of moving a car
    move_car_cost = 2

    actions = np.arange(-max_move_cars, max_move_cars + 1)

    # An up bound for poisson distribution
    # If n is greater than this value, then the probability of getting n is truncated to 0
    poisson_upper_bound = 11

    value = np.zeros((max_cars + 1, max_cars + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(max_cars + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # Policy evaluation (no actions used)
        while True:
            old_value = value.copy()
            for i in range(max_cars + 1):
                for j in range(max_cars + 1):
                    new_state_value = expected_return([i,j], policy[i,j], value, move_car_cost, max_cars, poisson_upper_bound, \
                                                      gamma, rental_credits, rental_request_first_loc, rental_request_second_loc,
                                                      returns_first_loc, returns_second_loc)
                    value[i, j] = new_state_value

            max_value_change = abs(old_value - value).max()
            if max_value_change < 1e-4:
                break
        
        # Policy Improvement
        policy_stable = True
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                old_action = policy[i,j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i,j], action, value, move_car_cost, max_cars, poisson_upper_bound, \
                                                                gamma, rental_credits, rental_request_first_loc, rental_request_second_loc,
                                                                returns_first_loc, returns_second_loc))
                    
                    else:
                        action_returns.append(-np.inf)
                
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action

                if policy_stable and old_action != new_action:
                    policy_stable = False

        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(max_cars + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break
            
        iterations += 1
    
    plt.show()
    plt.savefig("figure_4_2.png")
    plt.close()


def poisson_probability(n, lam):
    
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def expected_return(state, action, state_value, move_car_cost, max_cars, poisson_upper_bound, gamma, rental_credits,
                    rental_request_first_loc, rental_request_second_loc, returns_first_loc, returns_second_loc):
    """
    @states : [# of cars in first location, # of cars in second location]
    @action : positive if moving cars from first location to second location,
              negative if moving cars from second to first location. 
    @state_value : state_value_matrix
    """

    returns = 0

    # cost for moving cars
    returns -= move_car_cost * abs(action)

    # moving cars
    num_cars_first_loc = min(state[0] - action, max_cars)
    num_cars_second_loc = min(state[1] + action, max_cars)

    # go through all possible rental requests
    for rental_request_first_loc_ in range(poisson_upper_bound):
        for rental_request_second_loc_ in range(poisson_upper_bound):

            # probability for current combination of rental requests
            prob = poisson_probability(rental_request_first_loc_, rental_request_second_loc) * \
                   poisson_probability(rental_request_second_loc_, rental_request_second_loc)
            
            # valid rental requests should be less than actual # of cars
            valid_rental_first_loc = min(num_cars_first_loc, rental_request_second_loc_)
            valid_rental_second_loc = min(num_cars_second_loc, rental_request_second_loc_)

            # Get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * rental_credits
            
            num_of_cars_first_loc_ = num_cars_first_loc - valid_rental_first_loc
            num_of_cars_second_loc_ = num_cars_second_loc - valid_rental_second_loc

            for returned_cars_first_loc_ in range(poisson_upper_bound):
                for returned_cars_second_loc_ in range(poisson_upper_bound):

                    prob_return = poisson_probability(returned_cars_first_loc_, returns_first_loc) * \
                                  poisson_probability(returned_cars_second_loc_, returns_second_loc)
                    
                    num_cars_first_loc_2 = min(num_of_cars_first_loc_ + returned_cars_first_loc_, max_cars)
                    num_cars_second_loc_2 = min(num_of_cars_second_loc_ + returned_cars_second_loc_, max_cars)

                    prob_ = prob_return * prob
                    returns += prob_ * (reward + gamma * state_value[num_cars_first_loc_2, num_cars_second_loc_2])
    
    return returns



if __name__ == "__main__":
    # gamblers_problem()
    # gird_world()
    jacks_car_rental()



