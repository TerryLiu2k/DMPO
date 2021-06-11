import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, LogClient, LogServer, setSeed, collect, listStack
from ..utils import collectGraph
from ..models import MLP
from ..agents import MBPO, MultiAgent
from ..algorithm import RL
import ray

"""
    the hyperparameters are the same as MBPO, almost the same on Mujoco and Inverted Pendulum
"""
from algorithms.envs.NCS.real_net_env import RealNetEnv
import os
import configparser
import traci
import numpy as np

def radius2Adj(radius):
    config_path = "algorithms/envs/NCS/config/config_ma2c_nc_net.ini"
    config_path = os.path.join(os.path.dirname("."), config_path)
    config = configparser.ConfigParser()
    config.read(config_path)
    config = config['ENV_CONFIG']
    env = RealNetEnv(config)
    
    x = env.neighbor_mask
    x = x + x.transpose(1, 0) + np.eye(*x.shape, dtype="int")
    result = np.eye(*x.shape, dtype='int')
    for i in range(radius):
        result = result.dot(x)
    result = result.clip(0, 1)
    
    return result

def getArgs(radius_q, radius):
    # radius for p and pi

    adj = radius2Adj(radius)
    adj_q = radius2Adj(radius_q)
    degree = max(adj.sum(axis=1))
    degree_q = max(adj_q.sum(axis=1))
    
    gather = lambda x: collectGraph("gather", radius2Adj(x))
    reduce = lambda x: collectGraph("reduce", radius2Adj(x))

    algo_args = Config()
    algo_args.n_warmup=int(3e3) 
    """
     rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
     bs256lr3e-4, it takes 2e4updates
     for the model on CartPole to learn done...

     Only 3e5 samples are needed for parameterized input continous motion control (refer to MBPO)
     4e5 is needed fore model free CACC (refer to NeurComm)
    """
    algo_args.replay_size=int(1e6)
    algo_args.max_ep_len=720
    algo_args.test_interval = int(2e4)
    algo_args.batch_size=128 # MBPO used 256
    algo_args.n_step=int(1e8)
    algo_args.n_test = 5

    p_args=Config()
    p_args.network = MLP
    p_args.activation=torch.nn.ReLU
    p_args.lr=3e-4
    p_args.sizes = [22*degree, 64, 64, 64] 
    """
    SAC used 2 layers of width 256 for all experiments,
    MBPO used 4 layers of width 200 or 400
    NeurComm used 1 layer LSTM of width 64
    """
    p_args.update_interval=10
    p_args.update_interval_warmup = 1
    p_args.n_embedding = degree
    p_args.model_buffer_size = int(1e4)
    """
     bs=32 interval=4 from rainbow Q
     MBPO retrains fram scratch periodically
     in principle this can be arbitrarily frequent
    """
    p_args.n_p=3 # ensemble
    p_args.refresh_interval=int(1e3) # refreshes the model buffer
    # ideally rollouts should be used only once
    p_args.branch=1
    p_args.roll_length=1 # length > 1 not implemented yet
    p_args.to_predict = 'srd'

    q_args=Config()
    q_args.network = MLP
    q_args.activation=torch.nn.ReLU
    q_args.lr=3e-4
    q_args.sizes = [22*degree_q, 64, 64, 7] # 6 actions, dueling q learning
    q_args.update_interval=10
    # MBPO used 1/40 for continous control tasks
    # 1/20 for invert pendulum
    q_args.n_embedding = degree_q - 1

    pi_args=Config()
    pi_args.network = MLP
    pi_args.activation=torch.nn.ReLU
    pi_args.lr=3e-4
    pi_args.sizes = [22*degree, 64, 64, 6] 
    pi_args.update_interval=10

    agent_args=Config()
    pInWrapper = collect({'s': gather(radius), 'a': gather(radius), '*': gather(0)})
    #  (s, a) -> (s1, r, d), the ground truth for supervised training p
    qInWrapper = collect({'r':gather(0), 'd':gather(0), 'p_a1':gather(0), '*':gather(radius_q)})
    piInWrapper = collect({'s': gather(radius), 'q': reduce(radius_q)})
    wrappers = {'p_in': pInWrapper,
               'q_in': qInWrapper,
               'pi_in': piInWrapper}
    def MultiagentMBPO(**agent_args):
        agent_args['agent']=MBPO
        return MultiAgent(**agent_args)
    agent_args.wrappers = wrappers
    agent_args.agent=MultiagentMBPO
    agent_args.n_agent=28
    agent_args.gamma=0.99
    agent_args.alpha=0.2
    agent_args.target_entropy = 0.2
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate=5e-3
    # called tau in MBPO
    # sync rate per update = update interval/target sync interval

    agent_args.p_args = p_args
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    algo_args.agent_args = agent_args
        
    return algo_args