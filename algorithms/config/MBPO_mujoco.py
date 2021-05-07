import torch
import numpy as np
from ..utils import Config, Logger
from ..models import MLP
from ..agents import MBPO
from ..algorithm import RL
from .CartPole import env_name, env_fn

"""
    the hyperparameters are the same as MBPO, almost the same on Mujoco and Inverted Pendulum
"""
algo_args = Config()

algo_args.n_warmup=int(5e3)
"""
 rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
 bs256lr3e-4, it takes 2e4updates
 for the model on CartPole to learn done...

 Only 3e5 samples are needed for parameterized input continous motion control
"""
algo_args.replay_size=int(1e6)
algo_args.max_ep_len=500
algo_args.test_interval = int(1e4)
algo_args.seed=0
algo_args.batch_size=256 # the same as MBPO
algo_args.save_interval=600 # in seconds
algo_args.log_interval=int(2e3/200)
algo_args.n_step=int(1e8)

p_args=Config()
p_args.network = MLP
p_args.activation=torch.nn.ReLU
p_args.lr=3e-4
p_args.sizes = [4, 16, 32, 3] 
p_args.update_interval=1/10
"""
 bs=32 interval=4 from rainbow Q
 MBPO retrains fram scratch periodically
 in principle this can be arbitrarily frequent
"""
p_args.n_p=7 # ensemble
p_args.refresh_interval=int(1e3) # refreshes the model buffer
# ideally rollouts should be used only once
p_args.branch=400
p_args.roll_length=1 # length > 1 not implemented yet

q_args=Config()
q_args.network = MLP
q_args.activation=torch.nn.ReLU
q_args.lr=3e-4
q_args.sizes = [4, 16, 32, 3] # 2 actions, dueling q learning
q_args.update_interval=1/20
# MBPO used 1/40 for continous control tasks
# 1/20 for invert pendulum

pi_args=Config()
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=3e-4
pi_args.sizes = [4, 16, 32, 2] 
pi_args.update_interval=1/20

agent_args=Config()
agent_args.agent=MBPO
agent_args.gamma=0.99
agent_args.alpha=0.2 
agent_args.target_sync_rate=5e-3
# called tau in MBPO
# sync rate per update = update interval/target sync interval

args = Config()
args.env_name=env_name
args.name=f"{args.env_name}_{agent_args.agent}"
device = 0

q_args.env_fn = env_fn
agent_args.env_fn = env_fn
algo_args.env_fn = env_fn

agent_args.p_args = p_args
agent_args.q_args = q_args
agent_args.pi_args = pi_args
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set

print(f"rollout reuse:{(p_args.refresh_interval/q_args.update_interval*algo_args.batch_size)/algo_args.replay_size}")
# each generated data will be used so many times


RL(logger = Logger(args), device=device, **algo_args._toDict())