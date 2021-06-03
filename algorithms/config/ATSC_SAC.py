import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, LogClient, LogServer, setSeed, collect, listStack
from ..models import MLP
from ..utils import gather2D as _gather2D
from ..utils import reduce2D as _reduce2D
from ..agents import SAC, MultiAgent
from ..algorithm import RL
import ray

def getArgs(radius_q, radius):
    
    # radius for p and pi
    gather2D = lambda x: _gather2D((5, 5), x)
    reduce2D = lambda x: _reduce2D((5, 5), x)
    
    algo_args = Config()
    algo_args.n_warmup=0 
    """
     rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
     bs256lr3e-4, it takes 2e4updates
     for the model on CartPole to learn done...

     Only 3e5 samples are needed for parameterized input continous motion control (refer to MBPO)
     4e5 is needed fore model free CACC (refer to NeurComm)
    """
    algo_args.replay_size=int(1e6)
    algo_args.max_ep_len=720
    algo_args.test_interval = int(5e4)
    algo_args.batch_size=128
    algo_args.n_step=int(1e8)
    algo_args.n_test = 5

    q_args=Config()
    q_args.network = MLP
    q_args.activation=torch.nn.ReLU
    q_args.lr=3e-4
    q_args.sizes = [12*(1+2*radius_q)**2, 64, 64, 6] # 5 actions, dueling q learning
    q_args.update_interval=100
    # the same as SAC
    # MBPO used 1/40 for continous control tasks
    # 1/20 for invert pendulum
    q_args.n_embedding = (1+2*radius_q)**2 - 1

    pi_args=Config()
    pi_args.network = MLP
    pi_args.activation=torch.nn.ReLU
    pi_args.lr=3e-4
    pi_args.sizes = [12*(1+2*radius)**2, 64, 64, 5] 
    pi_args.update_interval=100

    agent_args=Config()
    qInWrapper = collect({'r':gather2D(0), 'd':gather2D(0), 'p_a1':gather2D(0), '*':gather2D(radius_q)})
    piInWrapper = collect({'s': gather2D(radius), 'q': reduce2D(radius_q)})
    wrappers = {'q_in': qInWrapper,
               'pi_in': piInWrapper}
    def MultiagentSAC(**agent_args):
        agent_args['agent']=SAC
        return MultiAgent(**agent_args)
    agent_args.wrappers = wrappers
    agent_args.agent=MultiagentSAC
    agent_args.n_agent=25
    agent_args.gamma=0.99
    agent_args.alpha=0.2
    agent_args.target_entropy = 0.2
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate=5e-3
    # called tau in MBPO
    # sync rate per update = update interval/target sync interval

    agent_args.p_args = None
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    algo_args.agent_args = agent_args
        
    return algo_args