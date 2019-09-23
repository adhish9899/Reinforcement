

import gym


env = gym.make("MountainCar-v0")
MAX_NUM_EPISODE = 5000

for episode in range(MAX_NUM_EPISODE):
    
    done = False
    obs = env.reset()
    totalReward = 0 # Keeping track record reward
    step = 0
    
    while not done:
        
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        totalReward += reward
        step += 1
        obs = next_state
        
    print("\n Episode #{} ended in {} steps. total_reward={}".format(episode, step+1,totalReward))

env.close()




