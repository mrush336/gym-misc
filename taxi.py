# link https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
import gym
import random
import numpy as np
from time import sleep
from IPython.display import clear_output

env = gym.make("Taxi-v3").env

state = env.encode(3, 1, 2, 0)
#print("State", state)
env.s = state

q_table = np.zeros([env.observation_space.n, env.action_space.n])
#print(q_table.shape)

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
frames = []

# function to display frames as the game progresses
def print_frame(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

# learning epochs
for i in range(1, 100):
    state = env.reset()  
    penalties, reward, epochs = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) # return values from the step()

        # update the q_table values
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1

        frames.append({'frame': env.render(mode='ansi'), 'state': state, 'action': action, 'reward': reward})

        
    print_frame(frames)
    
#print("Timesteps taken: {}".format(epochs))
#print("Penalties incurred: {}".format(penalties))

env.render()
#print("Action space {}".format(env.action_space))
#print("State space {}".format(env.observation_space))
print(q_table)