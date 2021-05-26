import gym
import numpy as np
from .NCS.cacc_env import CACCEnv
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
from ..utils import listStack

class CACCWrapper(gym.Wrapper):
    def __init__(self, config_path, bias=0, std=1):
        # k-hop
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        env = CACCEnv(config['ENV_CONFIG'])
        env.init_data(True, False, "/tmp")
        self.k = 0
        super().__init__(env)
        self.observation_space = Box(-1e6, 1e6, [(self.k*2+1)*5])
        self.action_space = Discrete(4)
        self.bias=bias
        self.std=std
    
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
    
    def reset(self):
        state = self.env.reset()
        state = np.array(state, dtype=np.float32)
        self.state = state
        return state
    
    def state2Reward(self, state):
        # accepts a (gpu) tensor
        reward, done =  self.env.state2Reward(state)
        return (reward+self.bias)/self.std, done
    
    def rescaleReward(self, acc_reward, episode_len):
        """
        acc_reward is sum over trajectory, mean over agent
        """
        acc_reward = acc_reward*self.std
        acc_reward -= episode_len*self.bias
        reward = acc_reward*8/episode_len
        return reward
        
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = np.array(state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        self.state=state
        """
        collision yields -1000*8, while the initial reward is -170, -1600 before collision
        I am worried if -8000 is a good solution compared with -170*(1/(1-gamma))...
        
        gets 1 if locally perfect, -4 and done if collision
        """
        return state, (reward+self.bias)/self.std, done, None
        

def CACC_catchup():
    return CACCWrapper('NCS/config/config_ma2c_nc_catchup.ini', bias=300, std=300)

def CACC_slowdown():
    return CACCWrapper('NCS/config/config_ma2c_nc_slowdown.ini', bias=300, std=300)