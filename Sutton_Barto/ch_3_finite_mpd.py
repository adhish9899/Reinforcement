
### GRID WORLD EXAMPLE MDP (SUTTON AND BARTO)

import numpy as np
import copy

def action_reward(intial_position, action, reward_size):

    if intial_position == [0,1]:
        return [4,1], 10

    if intial_position == [0,3]:
        return [2,3], 5

    final_position = np.array(intial_position) + np.array(action)

    if -1 in final_position or 5 in final_position:
        return intial_position, -1
    
    return final_position, reward_size


def gird_world():
    gamma = 0.9
    reward_size = 0
    grid_size = 5
    reward_states = [[0,1], [0,3]]
    actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    num_iterations = 100

    value_map = np.zeros((grid_size, grid_size))
    states = [[i, j] for i in range(grid_size) for j in range(grid_size)]

    deltas = []
    for it in range(num_iterations):
        copy_value_map = np.copy(value_map)
        delta_state =  []
        for state in states:
            weighted_rewards = 0
            for action in actions:
                final_position, reward = action_reward(state, action, reward_size)

                weighted_rewards += (1/len(actions))*(reward + gamma*value_map[final_position[0], final_position[1]])

            delta_state.append(np.abs(copy_value_map[state[0], state[1]] - weighted_rewards))
            copy_value_map[state[0], state[1]] = weighted_rewards
        
        deltas.append(delta_state)
        value_map = copy.deepcopy(copy_value_map)
        if it in [0,1,2,9, 99, num_iterations-1]:
            print("Iteration {}".format(it+1))
            print(np.round(value_map, 1))
            print("")


if __name__ == "__main__":
    gird_world()


