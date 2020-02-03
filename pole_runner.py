import numpy as np 
import gym
from neat import NEAT

env = gym.make('CartPole-v0')
    
def fitness_func(nn):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        env.render()
        # print(observation)
        action =  int(np.round(np.clip(nn.feed_forward(observation) , a_min=0 , a_max=1))[0])
        # action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            # print(total_reward)
            # print("Episode finished after {} timesteps".format(t+1))
            # total_reward /= t
            # print(total_reward)
            return total_reward 
    # total_reward /= 1000
    # print(total_reward)
    return total_reward

if __name__ == "__main__":
    print(env.action_space)
    print(env.observation_space)
    neat = NEAT(n_inputs=4 , n_outputs=1 , population_size=120)
    neat.train(fitness_func)
    env.close()