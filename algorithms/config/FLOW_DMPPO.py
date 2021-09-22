from gym.spaces.discrete import Discrete
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config

def getArgs(radius_p, radius_v, radius_pi, env):

    agent_args = Config()
    agent_args.adj = env.neighbor_mask
    agent_args.n_agent = agent_args.adj.shape[0]
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-5
    agent_args.lr_p = 5e-4
    agent_args.n_update_v = 10
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = False
    agent_args.advantage_norm = True
    agent_args.observation_space = env.observation_space
    agent_args.hidden_state_dim = 32
    agent_args.embedding_sizes = [env.observation_space.shape[0], 32, agent_args.hidden_state_dim]
    agent_args.observation_dim = agent_args.hidden_state_dim
    agent_args.action_space = env.action_space
    agent_args.adj = env.neighbor_mask
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = agent_args.action_space.n
    p_args.residual = True
    p_args.edge_embed_dim = 96
    p_args.node_embed_dim = 64
    p_args.edge_hidden_size = [64, 64]
    p_args.node_hidden_size = [64, 64]
    p_args.reward_coeff = 100.0
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

    alg_args = Config()
    alg_args.n_iter = 25000
    alg_args.n_warmup = 200
    alg_args.n_model_update = 10
    alg_args.n_model_update_warmup = 25
    alg_args.n_test = 5
    alg_args.test_interval = 10
    alg_args.rollout_length = 400
    alg_args.test_length = 400
    alg_args.max_episode_len = 400
    alg_args.model_based = True
    alg_args.n_traj = 1024
    alg_args.model_traj_length = 1
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = int(1e5)

    alg_args.agent_args = agent_args

    return alg_args
