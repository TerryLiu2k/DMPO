import gym
import numpy as np

class CartpoleWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        state = self.env.reset()
        self.state = np.array(state)
        return self.state
        
    def step(self, a):
        state, reward, done, info = self.env.step(a)
        self.state = np.array(state)
        return self.state, np.array(reward), np.array(done), None
    
env_name = 'CartPole-v1'
def CartPole():
    return CartpoleWrapper(gym.make(env_name))
env_fn = CartPole 

env = env_fn()
result  = np.array(env.reset())
print(result, result.dtype)