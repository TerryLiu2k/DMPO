import torch
import numpy as np
from ..utils import Config, Logger, setSeed, scatter, collect, listStack
from ..models import MLP
from ..agents import MBPO, MultiAgent
from ..algorithm import RL
from ..envs.CACC import env_fn

"""
    the hyperparameters are the same as MBPO, almost the same on Mujoco and Inverted Pendulum
"""
debug = True
radius = 1

algo_args = Config()
if getattr(algo_args, "checkpoint_dir", None) is None:
    algo_args.n_warmup=int(5e3)
else:
    algo_args.n_warmup=int(2.5e3) # enough for the model to fill the buffer
"""
 rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
 bs256lr3e-4, it takes 2e4updates
 for the model on CartPole to learn done...

 Only 3e5 samples are needed for parameterized input continous motion control
"""
if debug:
    algo_args.n_warmup=1
algo_args.replay_size=int(1e5)
if debug:
    algo_args.replay_size=1
algo_args.max_ep_len=600
if debug:
    algo_args.max_ep_len=2
algo_args.test_interval = int(1e3)
algo_args.batch_size=256 # the same as MBPO
algo_args.n_step=int(1e8)

p_args=Config()
p_args.network = MLP
p_args.activation=torch.nn.ReLU
p_args.lr=3e-4
p_args.sizes = [5*(1+2*radius), 32, 64] 
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
q_args.sizes = [5*(1+2*radius), 32, 64, 5] # 4 actions, dueling q learning
q_args.update_interval=1/20
# MBPO used 1/40 for continous control tasks
# 1/20 for invert pendulum

pi_args=Config()
pi_args.network = MLP
pi_args.activation=torch.nn.ReLU
pi_args.lr=3e-4
pi_args.sizes = [5*(1+2*radius), 32, 64, 4] 
pi_args.update_interval=1/20

pInWrapper = collect({'s': scatter(radius), 'a': scatter(radius), '*': scatter(0)})
#  (s, a) -> (s1, r, d), the ground truth for supervised training p
qWrapper = collect({'r':scatter(0), '*':scatter(radius)})
piInWrapper = collect({'s': scatter(1), 'q': scatter(radius)})

pOutWrapper = listStack
# (s, r, d)
piOutWrapper = lambda x: torch.stack(x, dim=1)
# (a)

wrappers = {'p_in': pInWrapper,
           'p_out': pOutWrapper,
           'q': qWrapper,
           'pi_in': piInWrapper,
           'pi_out': piOutWrapper}

agent_args=Config()
def MultiagentMBPO(**agent_args):
    agent_args['agent']=MBPO
    return MultiAgent(**agent_args)
agent_args.wrappers = wrappers
agent_args.agent=MultiagentMBPO
agent_args.n_agent=8
agent_args.gamma=0.99
agent_args.alpha=0.2 *0.2
agent_args.target_sync_rate=5e-3
# called tau in MBPO
# sync rate per update = update interval/target sync interval

args = Config()
device = 0
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed=0

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

setSeed(args.seed)
RL(logger = Logger(args, mute=debug), device=device, **algo_args._toDict()).run()