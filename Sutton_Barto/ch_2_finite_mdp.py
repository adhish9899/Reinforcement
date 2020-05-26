
### GRID WORLD EXAMPLE MDP (SUTTON AND BARTO)

import numpy as np

# Grid Size
grid = np.zeros((5,5))

# Gamma (Discount factor)
gamma = 0.9
iteration = 10

for num in range(iteration):
    for i in range(5): # all the rows
        for j in range(5): # all the columns
            
            up_grid = grid[i-1][j] if i > 0 else 0 # if going up takes us out of the grid then its value be 0
            down_grid = grid[i+1][j] if i < 4 else 0 # if going down takes us out of the grid then its value be 0
            left_grid = grid[i][j-1] if j > 0 else 0 # if going left takes us out of the grid then its value be 0
            right_grid = grid[i][j+1] if j < 4 else 0 # if going right takes us out of the grid then its value be 0

            all_dirs = [up_grid, down_grid, left_grid, right_grid]

            value=0
            if i == 0 and j == 1: # Position of A
                value = 10 + gamma*grid[4][1]
            
            elif i == 0 and j == 3: # the position of B
                value = 5 + gamma*grid[2][3]
            
            else:
                for direc in all_dirs:
                    if direc != 0:
                        value += 0.25 * (0 + gamma*direc) # if we don't go out of the grid
                    
                    else:
                        value += 0.25 * (-1 + gamma*grid[i][j]) # if we go out of the grid

            grid[i][j] = value


print("done")
print(np.round(grid, 1))
