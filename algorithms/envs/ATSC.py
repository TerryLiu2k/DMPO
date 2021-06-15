from .NCS.large_grid_env import LargeGridEnv
from .NCS.real_net_env import RealNetEnv
import os
import numpy as np
import gym
import configparser

class ATSCWrapper(gym.Wrapper):
    def __init__(self, config_path, n_agent,):
        # k-hop
        self.n_agent = n_agent
        config_path = os.path.join(os.path.dirname("."), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config['ENV_CONFIG']
        if self.n_agent == 28:
            env = RealNetEnv(config)
            phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
        else:
            env = LargeGridEnv(config)
            self.n_action = [4]*25
        super().__init__(env)
        
    def reset(self):
        state = self.env.reset()
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        self.state = state
        return state    
    
    def rescaleReward(self, reward, _):
        return reward*200/720*self.n_agent
        
    def step(self, action):
        """
        reward scaling is necessary since SAC temperature tuning can be slow to adapt to large reward
        """
        if self.n_agent == 28:
            for i in range(len(action)):
                if action[i]>= self.n_action[i]:
                    action[i] = np.random.randint(self.n_action[i])
                
        state, reward, done, info = self.env.step(action)
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*self.n_agent, dtype=np.float32)
        self.state=state
        return state, reward/200, done, None

    def get_state(self):
        return self.state
    
def ATSCGrid():
    return ATSCWrapper("algorithms/envs/NCS/config/config_ma2c_nc_grid.ini", 25)

def ATSCNet():
    return ATSCWrapper("algorithms/envs/NCS/config/config_ma2c_nc_net.ini", 28)
