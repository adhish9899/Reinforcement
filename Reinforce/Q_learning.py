


import gym
import numpy as np

episilon_min = 0.005
max_num_steps = max_num_episode * steps_per_episode
episilon_decay = 500 * episilon_min / max_num_steps
alpha = 0.05 # Learning rate
gamma = 0.98
num_discrete_steps = 30 # Number of bins to Discretize each observation dim

def q_learner(object):
    
    def __init__(self,env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = num_discrete_steps
        self.bin_width = (self.obs_high - self.obs_low)/self.obs_bin
        self.action_shape = env.action_space.n

        # Create a multi-dimensional array (aka. Table) to represent the Q-values
        
        self.Q = np.zeros((self.obs_bin + 1, self.obs_bin + 1, self.action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.episilon = 1.0
        
    
    def discretize(self,obs):

        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))
    
    
    def get_action(self,obs):
        
        discretized_obs = self.discretize(obs)
        
        # Epsilon-Greedy action selection
        if self.epsilon > episilon_min:
            self.epsilon -= episilon_decay
            
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        
        else: # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])
        
        
        
        







