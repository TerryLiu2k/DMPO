import torch
import configparser, time
from ..utils import Config, collect
from ..utils import gather2D as _gather2D
from ..utils import reduce2D as _reduce2D
from ..agents import NeurCommWrapper

def getArgs(radius_p, radius_q, radius_pi, env):
    gather2D = lambda x: _gather2D((3, 3), x)
    reduce2D = lambda x: _reduce2D((3, 3), x)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.n

    algo_args = Config()
    algo_args.n_warmup = 0
    algo_args.replay_size = int(1e6)
    algo_args.imm_size = algo_args.replay_size
    algo_args.max_ep_len = 720
    algo_args.test_interval = int(4e3)
    algo_args.batch_size = 128  # MBPO used 256
    algo_args.n_step = int(1e8)
    algo_args.n_test = 5

    p_args, q_args, pi_args = None, None, Config()

    agent_args = Config()
    agent_config = configparser.ConfigParser()
    agent_config.read('./algorithms/config/config_FLOW_NC.ini')
    agent_args.agent_config = agent_config

    algo_args.agent_args = agent_args
    agent_args.p_args = p_args
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args

    agent_args.agent = NeurCommWrapper
    agent_args.n_agent = 9
    agent_args.gamma = 0.99
    agent_args.alpha = 0
    agent_args.target_entropy = None
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate = 5e-3

    return algo_args


