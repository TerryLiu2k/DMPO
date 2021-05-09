import torch
import numpy as np
from ..utils import Config, Logger, setSeed
from ..models import CNN
from ..agents import SAC
from ..algorithm import RL
from ..envs.Breakout import env_name, env_fn

"""
    Compared with QLearning, alpha instead of eps
    notice that 50M samples is typical for DQNs with visual input (refer to rainbow)
    Hyperparameters are more aggressive compared with rainbow
    the batchsize, lr and tau is much higher, smaller replay buffer shorter warnmup
"""
algo_args = Config()

algo_args.max_ep_len=2000
algo_args.batch_size=256
algo_args.n_warmup=int(3e4)
# refer to rainbow 80K, while 200K is typical for Q learning with less trick
algo_args.replay_size=int(2e5)
#  rainbow used 1e6
algo_args.test_interval = int(3e4)
algo_args.n_step=int(1e8)

q_args=Config()
q_args.network = CNN
q_args.update_interval=4
q_args.activation=torch.nn.ReLU
q_args.lr=2e-4
q_args.strides = [2]*6
q_args.kernels = [3]*6
q_args.paddings = [1]*6
q_args.sizes = [4, 16, 32, 64, 128, 128, 5] # 4 actions, dueling q learning

pi_args=Config()
pi_args.update_interval=4
pi_args.network = CNN
pi_args.activation=torch.nn.ReLU
pi_args.lr=2e-4
pi_args.strides = [2]*6
pi_args.kernels = [3]*6
pi_args.paddings = [1]*6
pi_args.sizes = [4, 16, 32, 64, 128, 128, 4] 

agent_args=Config()
agent_args.agent=SAC
agent_args.gamma=0.99
agent_args.alpha=0.2/50 # 50 frames 1 reward is typical
agent_args.target_sync_rate=2/1000
# also called tau

args = Config()
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed=0
device = 0

q_args.env_fn = env_fn
agent_args.env_fn = env_fn
algo_args.env_fn = env_fn

agent_args.p_args = None
agent_args.q_args = q_args
agent_args.pi_args = pi_args
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set

setSeed(args.seed)
RL(logger = Logger(args), device=device, **algo_args._toDict()).run()