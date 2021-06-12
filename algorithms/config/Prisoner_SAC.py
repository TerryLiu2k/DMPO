import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, LogClient, LogServer, setSeed, collect, gather, reduce, listStack
from ..models import MLP
from ..agents import SAC, MultiAgent
from ..algorithm import RL
import ray

def getArgs(env, radius_q, radius_p, radius_pi):
    observation_dim = env.observation_space.shape[0]
    n_action = env.action_space.n
    
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
    algo_args.max_ep_len=5
    algo_args.test_interval = int(2e3)
    algo_args.batch_size=128
    algo_args.n_step=int(1e8)
    algo_args.n_test = 10

    q_args=Config()
    q_args.network = MLP
    q_args.activation=torch.nn.ReLU
    q_args.lr=3e-4
    q_args.sizes = [observation_dim*(1+2*radius_q), 64, 64, n_action+1] # 5 actions, dueling q learning
    q_args.update_interval=1
    # the same as SAC
    # MBPO used 1/40 for continous control tasks
    # 1/20 for invert pendulum
    q_args.n_embedding = max((1+2*radius_q) - 1, 0)

    pi_args=Config()
    pi_args.network = MLP
    pi_args.activation=torch.nn.ReLU
    pi_args.lr=3e-4
    pi_args.sizes = [observation_dim*(1+2*radius_pi), 64, 64, n_action] 
    pi_args.update_interval=1

    agent_args=Config()
    qInWrapper = collect({'p_a1':gather(0), 'd': gather(0), 'r': reduce(radius_q) ,'*':gather(radius_q)})
    # s, a, r, s1, a1, p_a1, d
    piInWrapper = collect({'s': gather(radius_pi), 'q': gather(0)})
    wrappers = {'q_in': qInWrapper,
               'pi_in': piInWrapper}
    def MultiagentSAC(**agent_args):
        agent_args['agent']=SAC
        return MultiAgent(**agent_args)
    agent_args.wrappers = wrappers
    agent_args.agent=MultiagentSAC
    agent_args.n_agent=5
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