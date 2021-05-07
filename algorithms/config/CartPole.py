import gym
import numpy as np

class CartpoleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, x):
        x = np.array(x, dtype=np.float32)
        return x
    
env_name = 'CartPole-v1'
env_fn = lambda: CartpoleWrapper(gym.make(env_name))

env = env_fn()
result  = np.array(env.reset())
print(result, result.dtype)