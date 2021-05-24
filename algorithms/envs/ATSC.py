from .NCS.large_grid_env import LargeGridEnv
import os
import configparser

class ATSCWrapper(gym.Wrapper):
    def __init__(self, config_path):
        # k-hop
        config_path = "algorithms/envs/NCS/config/config_ma2c_nc_grid.ini"
        config_path = os.path.join(os.path.dirname("."), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config['ENV_CONFIG']
        env = LargeGridEnv(config)
        super().__init__(env)
        self.observation_space = Box(0, 1e6, [12])
        self.action_space = Discrete(5)
    
def ATSCGrid():
    return ATSCWrapper()