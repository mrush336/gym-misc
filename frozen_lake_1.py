# link https://deeplizard.com/learn/video/QK_PP_2KgGE
import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# setup the env
env = gym.make("FrozenLake8x8-v0", is_slippery=False)
observation = env.reset()

# setup the q-table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
#print(q_table)

# instaniate global variables
num_episodes = 10000
steps_per_episodes = 1000
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = .1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# empty list to hold our rewards over time
rewards_all_episodes = []
 
 # main loops
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(steps_per_episodes):
        
        # exploration vs exploitation
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        #print(next_state)
        #print(q_table.shape)

        # update q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[next_state, :]))

        state = next_state
        rewards_current_episode += reward
        
        if done == True:
            break
        
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)
