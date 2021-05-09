import torch
import numpy as np
from ..utils import Config, Logger
from ..models import MLP
from ..agents import SAC, MultiAgent
from ..algorithm import RL
from ..envs.CACC import env_fn

"""
    Compared with QLearning, alpha instead of eps
    notice that 50M samples is typical for DQNs with visual input (refer to rainbow)
"""


algo_args = Config()
debug = False
neighbor_radius=1

algo_args.max_ep_len=600
if debug:
    algo_args.max_ep_len=2
algo_args.batch_size=256
algo_args.n_warmup=int(5e3)
if debug:
    algo_args.n_warmup=1
algo_args.replay_size=int(1e5)
# high replay size slows down training a lot
# since new samples are less frequently sampled
algo_args.test_interval = int(1e3)
algo_args.n_step=int(1e8)
algo_args.neighbor_radius = neighbor_radius

q_args=Config()
q_args.network = MLP
q_args.update_interval=1
q_args.activation=torch.nn.ReLU
q_args.lr=3e-4
q_args.sizes = [5*(1+2*neighbor_radius), 32, 64, 5] 

pi_args=Config()
pi_args.update_interval=1
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=3e-4
pi_args.sizes = [5*(1+2*neighbor_radius), 32, 64, 4] 

agent_args=Config()
def agent_fn(**agent_args):
    agent_args['agent']=SAC
    return MultiAgent(**agent_args)
agent_args.agent=agent_fn
agent_args.n_agent=8
agent_args.gamma=0.99
agent_args.alpha=0.2 *0.2 # reward rescaled
agent_args.target_sync_rate=5e-3
# rainbow used 32K samples per q target sync
# high sync rate causes q becomes nan 

args = Config()
device = 0
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed = 0

q_args.env_fn = env_fn(algo_args.neighbor_radius)
agent_args.env_fn = env_fn(algo_args.neighbor_radius)
algo_args.env_fn = env_fn(algo_args.neighbor_radius)

agent_args.p_args = None
agent_args.q_args = q_args
agent_args.pi_args = pi_args
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set

setSeed(args.seed)
RL(logger = Logger(args, mute=debug), device=device, **algo_args._toDict()).run()