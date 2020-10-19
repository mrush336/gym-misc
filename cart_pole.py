import gym

env = gym.make("CartPole-v1")
observation = env.reset()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished in {} timesteps".format(t+1))
            break

env.close()