


import gym
import numpy as np

max_num_episode = 50000
steps_per_episode = 200 # This is specific to MountainCar. May change with env

episilon_min = 0.005
max_num_steps = max_num_episode * steps_per_episode
episilon_decay = 500 * episilon_min / max_num_steps
alpha = 0.05 # Learning rate
gamma = 0.98
num_discrete_steps = 30 # Number of bins to Discretize each observation dim

class q_learner(object):
    
    def __init__(self,env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bin = num_discrete_steps
        self.bin_width = (self.obs_high - self.obs_low)/self.obs_bin
        self.action_shape = env.action_space.n

        # Create a multi-dimensional array (aka. Table) to represent the Q-values
        
        self.Q = np.zeros((self.obs_bin + 1, self.obs_bin + 1, self.action_shape))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0

    
    def discretize(self,obs):
        
        ## Bin of the current observation
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
        
    
    def learn(self,obs,action,reward,next_obs):
        
        discretized_obs = self.discretize(obs)
        
        discretized_next_obs = self.discretize(next_obs)
        
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        
        td_error = td_target - self.Q[discretized_obs][action] # Reward by taking that particular action
        
        self.Q[discretized_obs][action] += self.alpha * td_error
        

def train(agent,env):
    
    best_reward = -float('inf')
    
    for episode in range(max_num_episode):
        
        done = False
        obs = env.reset()
        total_reward = 0.0
        
        while not done:
            
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs,action,reward,next_obs)
            obs = next_obs
            total_reward += reward
            
        if total_reward > best_reward:
            best_reward = total_reward
        
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
              total_reward, best_reward, agent.epsilon))
    
    
    return np.argmax(agent.Q, axis=2)


def test(agent,env,policy):
    
    done = False
    obs = env.reset()
    total_reward = 0.0
    
    while not done:
        
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
        
    return total_reward
        


if __name__ == "__main__":
    
    env = gym.make('MountainCar-v0')
    agent = q_learner(env)
    learned_policy = train(agent, env)
    
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()




