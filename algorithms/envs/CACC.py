import gym
import numpy as np
from .NCS.cacc_env import CACCEnv
import configparser
import os
import pdb

class CACCWrapper(gym.Wrapper):
    def __init__(self, config_path, k):
        # k-hop
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        env = CACCEnv(config['ENV_CONFIG'])
        env.init_data(True, False, "/tmp")
        self.k = k
        super().__init__(env)
    
    def ifCollide(self):
        ob = self.state
        normalized_v = np.array([item[0] for item in ob])
        normalized_h = np.array([item[3] for item in ob])
        v = normalized_v * self.v_star + self.v_star
        v = np.concatenate((np.array([self.v_star]), v), axis=0)
        h = normalized_h * self.h_star + self.h_star
        h = h - (v[:-1]-v[1:])*self.dt
        if np.min(h) < self.h_min:
            return True
        return False
    
    def _k_hop(self, state):
        result = np.zeros((8, 5*(1+2*self.k)), dtype = np.float32)
        for i in range(8):
            for j in range(i-self.k, i+self.k+1):
                if j<0 or j>=8:
                    continue
                start = j-(i-self.k)
                result[i, start*5: start*5+5] = state[j]
        return result
    
    def reset(self):
        state = self.env.reset()
        state = np.array(state, dtype=np.float32)
        state = self._k_hop(state)
        self.state = state
        return state
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = np.array(state, dtype=np.float32)
        reward = np.array([reward]*8, dtype=np.float32)
        done = np.array([done]*8, dtype=np.float32)
        state = self._k_hop(state)
        self.state=state
        return state, reward/1000, done, None
        

env_name = 'CACC_catchup'
env_fn = lambda k: (lambda: CACCWrapper('NCS/config/config_ma2c_nc_catchup.ini', k))

env = env_fn(1)()
result  = np.array(env.reset())
print(result, result.dtype)
