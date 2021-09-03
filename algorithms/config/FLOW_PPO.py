from gym.spaces.discrete import Discrete
from numpy import pi
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config
from algorithms.mbdppo.MB_DPPO import MB_DPPOAgent

def getArgs(radius_p, radius_v, radius_pi, env):

    alg_args = Config()
    alg_args.n_iter = 25000
    alg_args.n_warmup = 6
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 5
    alg_args.test_interval = 5
    alg_args.rollout_length = 400
    alg_args.test_length = 400
    alg_args.max_episode_len = 400
    alg_args.model_based = False
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0

    agent_args = Config()
    agent_args.adj = env.neighbor_mask
    agent_args.n_agent = agent_args.adj.shape[0]
    agent_args.gamma = 0.99
    agent_args.lamda = 0.95
    agent_args.clip = 0.4
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-4
    agent_args.n_update_v = 10
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.advantage_norm = True
    agent_args.observation_space = env.observation_space
    agent_args.action_space = env.action_space
    agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p

    p_args = None
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 64, 64, 1]
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    action_dim = env.action_space.n
    pi_args.sizes = [-1, 64, 64, action_dim]
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args