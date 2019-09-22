

'''
import gym

## Making an enviornment
env = gym.make('Alien-ram-v0')

## Get the first observation of the enviornment 
obs = env.reset()

# Inner Loop
action = agent.choose_action(obs)
next_state,reward,done,info = env.step(action)
obs = next_state
## Repeat Inner Loop
'''

import gym

env = gym.make("Qbert-v0")

MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    
    obs = env.reset()
    
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        obs = next_state
        
        if done is True:
            print("\n Episode #{} ended in {} step.".format(episode,step + 1))
            break
        



