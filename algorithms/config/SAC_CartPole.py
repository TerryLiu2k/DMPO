import torch
import numpy as np
from ..utils import Config, Logger, setSeed
from ..models import MLP
from ..agents import SAC
from ..algorithm import RL
from ..envs.CartPole import env_name, env_fn

"""
    Compared with QLearning, alpha instead of eps
    notice that 50M samples is typical for DQNs with visual input (refer to rainbow)
"""


algo_args = Config()

algo_args.max_ep_len=2000
algo_args.batch_size=256
algo_args.n_warmup=int(2e5)
algo_args.replay_size=int(1e5)
# high replay size slows down training a lot
# since new samples are less frequently sampled
algo_args.test_interval = int(3e4)
algo_args.n_step=int(1e8)

q_args=Config()
q_args.network = MLP
q_args.update_interval=32
q_args.activation=torch.nn.ReLU
q_args.lr=2e-4
q_args.sizes = [4, 16, 32, 3] # 2 actions, dueling q learning

pi_args=Config()
pi_args.update_interval=32
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=2e-4
pi_args.sizes = [4, 16, 32, 2] 

agent_args=Config()
agent_args.agent=SAC
agent_args.gamma=0.99
agent_args.alpha=0.2 
agent_args.target_sync_rate=5e-3
# rainbow used 32K samples per q target sync
# high sync rate causes q becomes nan 

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