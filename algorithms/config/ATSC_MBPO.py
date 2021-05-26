import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, LogClient, LogServer, setSeed, collect, listStack
from ..utils import gather2D as _gather2D
from ..utils import reduce2D as _reduce2D
from ..models import MLP
from ..agents import MBPO, MultiAgent
from ..algorithm import RL
import ray

"""
    the hyperparameters are the same as MBPO, almost the same on Mujoco and Inverted Pendulum
"""


def main(env_fn, debug=False, test=False, seed=None, init_checkpoint=None):
    
    radius_q = 2
    radius = 1
    # radius for p and pi
    gather2D = lambda x: _gather2D((5, 5), x)
    reduce2D = lambda x: _reduce2D((5, 5), x)

    algo_args = Config()
    algo_args.n_warmup=int(2.5e3) # enough for the model to fill the buffer
    """
     rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
     bs256lr3e-4, it takes 2e4updates
     for the model on CartPole to learn done...

     Only 3e5 samples are needed for parameterized input continous motion control (refer to MBPO)
     4e5 is needed fore model free CACC (refer to NeurComm)
    """
    algo_args.replay_size=int(1e6)
    algo_args.max_ep_len=720
    algo_args.test_interval = int(1e3)
    algo_args.batch_size=1 # MBPO used 256
    algo_args.n_step=int(1e8)
    algo_args.n_test = 10
    algo_args.init_checkpoint = init_checkpoint
    if debug:
        algo_args.batch_size = 4
        algo_args.max_ep_len=2
        algo_args.replay_size=1
        algo_args.n_warmup=1
    if test:
        algo_args.n_warmup = 0
        algo_args.n_test = 50

    p_args=Config()
    p_args.network = MLP
    p_args.activation=torch.nn.ReLU
    p_args.lr=3e-4
    p_args.sizes = [12*(1+2*radius_q)**2, 64, 64, 64] 
    """
    SAC used 2 layers of width 256 for all experiments,
    MBPO used 4 layers of width 200 or 400
    NeurComm used 1 layer LSTM of width 64
    """
    p_args.update_interval=1/10
    p_args.n_embedding = (1+2*radius)
    """
     bs=32 interval=4 from rainbow Q
     MBPO retrains fram scratch periodically
     in principle this can be arbitrarily frequent
    """
    p_args.n_p=7 # ensemble
    p_args.refresh_interval=int(1e3) # refreshes the model buffer
    # ideally rollouts should be used only once
    p_args.branch=1
    p_args.roll_length=1 # length > 1 not implemented yet
    p_args.to_predict = 's'

    q_args=Config()
    q_args.network = MLP
    q_args.activation=torch.nn.ReLU
    q_args.lr=3e-4
    q_args.sizes = [12*(1+2*radius_q)**2, 64, 64, 5] # 4 actions, dueling q learning
    q_args.update_interval=1/5
    # MBPO used 1/40 for continous control tasks
    # 1/20 for invert pendulum
    q_args.n_embedding = (2*radius_q)

    pi_args=Config()
    pi_args.network = MLP
    pi_args.activation=torch.nn.ReLU
    pi_args.lr=3e-4
    pi_args.sizes = [12*(1+2*radius)**2, 64, 64, 4] 
    pi_args.update_interval=1/5

    agent_args=Config()
    pInWrapper = collect({'s': gather2D(radius), 'a': gather2D(radius), '*': gather2D(0)})
    #  (s, a) -> (s1, r, d), the ground truth for supervised training p
    qInWrapper = collect({'r':gather2D(0), 'd':gather2D(0), 'p_a1':gather2D(0), '*':gather2D(radius_q)})
    piInWrapper = collect({'s': gather2D(radius), 'q': reduce2D(radius_q)})
    wrappers = {'p_in': pInWrapper,
               'q_in': qInWrapper,
               'pi_in': piInWrapper}
    def MultiagentMBPO(**agent_args):
        agent_args['agent']=MBPO
        return MultiAgent(**agent_args)
    agent_args.wrappers = wrappers
    agent_args.agent=MultiagentMBPO
    agent_args.n_agent=25
    agent_args.gamma=0.99
    agent_args.alpha=0.2
    agent_args.target_entropy = 0.2
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate=1e-3
    # called tau in MBPO
    # sync rate per update = update interval/target sync interval

    args = Config()
    args.save_period=1800 # in seconds
    args.log_period=int(20)
    if  seed is None:
        seed = np.random.randint(65536)
    args.seed = seed
    args.test = test

    algo_args.env_fn = env_fn
    agent_args.p_args = p_args
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    algo_args.agent_args = agent_args
    args.algo_args = algo_args # do not call toDict() before config is set
    algo_args.seed = args.seed
        
    print(f"rollout reuse:{(p_args.refresh_interval/q_args.update_interval*algo_args.batch_size)/algo_args.replay_size}")
    # each generated data will be used so many times
    ray.init(ignore_reinit_error = True, num_gpus=2)
    logger = LogServer.remote(args, mute=debug or test)
    logger = LogClient(logger)
    RL(logger = logger, **algo_args._toDict()).run()