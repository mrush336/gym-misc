import gym

env = gym.make("FrozenLake8x8-v0", is_slippery=False)
observation = env.reset()

for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Reward: {}".format(reward))
            print(info)
            print("Episode finished in {} timesteps".format(t+1))
            break

env.close()