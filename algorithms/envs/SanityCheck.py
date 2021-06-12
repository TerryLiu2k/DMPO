import numpy as np
from gym.spaces import Box, Discrete
class _Prisoner():
    """
    Prisoner's delimma
    action 1: self + 1, others -1
    """
    def __init__(self, n):
        self.n = n
        self.state = np.zeros((n, 1), dtype='float32')
        self.action_space = Discrete(2)
        self.observation_space = Box(-1, 1, shape=[1])
        
    
    def reset(self):
        return self.state
        
    def step(self, action):
        reward = action * 2 - action.sum()
        return self.state, reward, np.ones((self.n), dtype='int32'), None
    
def Prisoner(x):
    def NAgentPrisoner():
        return _Prisoner(x)
    return NAgentPrisoner