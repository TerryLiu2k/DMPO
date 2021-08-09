import torch
from ..utils import Config, collect
from ..utils import gather2D as _gather2D
from ..utils import reduce2D as _reduce2D
from ..models import MLP
from ..agents import SAC_New, MultiAgent

def getArgs(radius_q, radius_p, radius_pi, env):
    gather2D = lambda x: _gather2D((5, 5), x)
    reduce2D = lambda x: _reduce2D((5, 5), x)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.n

    algo_args = Config()
    algo_args.n_warmup = 0
    algo_args.replay_size=int(1e6)
    algo_args.imm_size = 7200
    algo_args.max_ep_len = 720
    algo_args.test_interval = int(4e3)
    algo_args.batch_size = 128
    algo_args.n_step = int(1e8)
    algo_args.n_test = 5

    p_args = None

    q_args = Config()
    q_args.network = MLP
    q_args.activation = torch.nn.ReLU
    q_args.lr = 3e-4
    q_args.n_embedding = 5
    q_args.sizes = [(obs_dim + q_args.n_embedding) * (1 + 2 * radius_q) ** 2, 64, 64, 1]
    q_args.update_interval = 4
    q_args.update_steps = 1

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.lr = 3e-4
    pi_args.sizes = [obs_dim * (1 + 2 * radius_pi) ** 2, 64, 64, action_dim]
    pi_args.update_interval = 20
    pi_args.update_steps = 1

    agent_args = Config()
    pInWrapper = collect({'s': gather2D(radius_p), 'a': gather2D(radius_p), '*': gather2D(0)})
    #  (s, a) -> (s1, r, d), the ground truth for supervised training p
    qInWrapper = collect({'p_a1': gather2D(0), 'd': gather2D(0), 'r': gather2D(0), '*': gather2D(radius_q)})
    # s, a, r, s1, a1, p_a1, d
    piInWrapper = collect({'s': gather2D(radius_pi), 'q': gather2D(0)})
    qOutWrapper = lambda x: reduce2D(radius_q)(torch.stack(x, dim=1))
    wrappers = {'p_in': pInWrapper,
                'q_in': qInWrapper,
                'pi_in': piInWrapper,
                'q_out': qOutWrapper}

    def MultiagentSAC(**agent_args):
        agent_args['agent'] = SAC_New
        return MultiAgent(**agent_args)

    agent_args.wrappers = wrappers
    agent_args.agent = MultiagentSAC
    agent_args.n_agent = 25
    agent_args.gamma = 0.99
    agent_args.alpha = 0
    agent_args.target_entropy = None
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate = 5e-3
    # called tau in MBPO
    # sync rate per update = update interval/target sync interval

    agent_args.p_args = p_args
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    algo_args.agent_args = agent_args

    return algo_args
