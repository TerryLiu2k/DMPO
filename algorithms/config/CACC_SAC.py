import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, Logger, setSeed, gather, collect, listStack, reduce
from ..models import MLP
from ..agents import SAC, MultiAgent
from ..algorithm import RL
from ..envs.CACC import env_fn

"""
    the hyperparameters are the same as MBPO, almost the same on Mujoco and Inverted Pendulum
"""
debug = False
radius = 2
radius_pi = 2

algo_args = Config()
algo_args.n_warmup=int(2e3) # enough for the model to fill the buffer
"""
 rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
 bs256lr3e-4, it takes 2e4updates
 for the model on CartPole to learn done...

 Only 3e5 samples are needed for parameterized input continous motion control
 Only 3e4 needed for CACC
"""
algo_args.replay_size=int(1e5)
algo_args.max_ep_len=600
algo_args.test_interval = int(1e3)
algo_args.batch_size=256 # the same as MBPO
algo_args.n_step=int(1e8)
if debug:
    algo_args.batch_size = 4
    algo_args.max_ep_len=2
    algo_args.replay_size=1
    algo_args.n_warmup=1

q_args=Config()
q_args.network = MLP
q_args.activation=torch.nn.ReLU
q_args.lr=3e-4
q_args.sizes = [5*(1+2*radius), 32, 64, 5] # 4 actions, dueling q learning
q_args.update_interval=1/20
# MBPO used 1/40 for continous control tasks
# 1/20 for invert pendulum
q_args.n_embedding = (2*radius)

pi_args=Config()
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=3e-4
pi_args.sizes = [5*(1+2*radius_pi), 32, 64, 4] 
pi_args.update_interval=1/20

qInWrapper = collect({'r':gather(0), 'd':gather(0), 's1': gather(radius), '*':gather(radius)})
piInWrapper = collect({'s': gather(radius_pi), 'q': reduce(radius)})

wrappers = {'q_in': qInWrapper,
           'pi_in': piInWrapper}

agent_args=Config()
def MultiagentSAC(**agent_args):
    agent_args['agent']=SAC
    return MultiAgent(**agent_args)
agent_args.wrappers = wrappers
agent_args.agent=MultiagentSAC
agent_args.n_agent=8
agent_args.gamma=0.99
agent_args.alpha=0.2
agent_args.target_entropy = 0.2
# 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
agent_args.target_sync_rate=5e-3
# called tau in MBPO
# sync rate per update = update interval/target sync interval

args = Config()
device = 0
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed=np.random.randint(65536)

q_args.env_fn = env_fn
agent_args.env_fn = env_fn
algo_args.env_fn = env_fn

agent_args.q_args = q_args
agent_args.pi_args = pi_args
agent_args.p_args = None
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set

setSeed(args.seed)
RL(logger = Logger(args, mute=debug), device=device, **algo_args._toDict()).run()