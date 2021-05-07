from gym.wrappers import FrameStack

class BreakoutWrapper(gym.ObservationWrapper):
    """ 
    takes (210, 160, 3) to (40, 40)
    stops training when one life is lost
    converts to grey scale float
    cuts the margins
    
    fires the ball by pressing action 1
    
    wrapped by framestack (not wrapping FrameStack) to utilize lazy frame for memory saving
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(0, 1, (170, 160))
        self.pooling = torch.nn.AvgPool2d(kernel_size=(4,4), stride=(4, 4))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            done = True
        self.lives = lives
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        obs, _, _, _ = self.step(1)
        return obs

    def observation(self, observation):
        observation = np.array(observation).astype(np.float32) / 255.0
        observation = observation[30:-17] 
        observation = np.mean(observation, axis=2) # greyscale
        tmp = torch.as_tensor(observation).unsqueeze(0)
        observation = np.array(self.pooling(tmp).squeeze(0))
        return observation

from agents import QLearning
from matplotlib import pyplot as plt
env_name = 'Breakout-v0'
env_fn = lambda: FrameStack(BreakoutWrapper(gym.make(env_name)), 4)

env = env_fn()
result  = np.array(env.reset())
result = np.array(result).transpose(1, 2, 0) # 0 is white
#plt.imshow(result[:, :, -1], cmap='Greys') 
plt.imshow(1-result[:, :, 1:4]) 