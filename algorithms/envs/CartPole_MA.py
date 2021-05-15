import gym
import numpy as np
from numpy import random

class CartpoleWrapper(gym.Wrapper):
    """
    Cartpole using stochastic agent ensemble
    for debugging multi agent RL
    """
    def __init__(self, env, n_agent=2):
        gym.Wrapper.__init__(self, env)
        self.n_agent = n_agent

    def reset(self):
        state = self.env.reset()
        self.state = np.array([state]*self.n_agent)
        return self.state
        
    def step(self, a):
        n_agent = self.n_agent
        a = a[random.randint(n_agent)]
        state, reward, done, info = self.env.step(a)
        self.state = np.array([state]*n_agent)
        return self.state, np.array([reward]*n_agent), np.array([done]*n_agent), None
    
env_name = 'CartPole-v1'
def CartPole():
    return CartpoleWrapper(gym.make(env_name))
env_fn = CartPole

env = env_fn()
result  = np.array(env.reset())
print(result, result.dtype)