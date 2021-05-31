import os
import numpy as np
import ray
from algorithms.utils import Config, LogClient, LogServer
from algorithms.algorithm import RL

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'

"""
This section contains run args, separated from args for the RL algorithm and agents
"""
args = Config()
#### computation
os.environ['CUDA_VISIBLE_DEVICES']='1'
args.n_thread = 1
args.parallel = False
args.device = 'cpu'
args.n_cpu = 1/2 # per agent, used only if parallel = True
args.n_gpu = 0

#### general
args.debug = False
args.test = False # if no training, only test
args.profiling = False

#### algorithm and environment
from algorithms.config.ATSC_MBPO import getArgs
#from algorithms.envs.CACC import CACC_catchup as env_fn
#from algorithms.envs.CACC import CACC_slowdown as env_fn
from algorithms.envs.ATSC import ATSCGrid as env_fn
args.name = 'large_radius'
args.radius_q=3
args.radius=3

#### checkpoint
args.init_checkpoint = None
args.start_step = 0

#### misc
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed = None

algo_args = getArgs(radius_q=args.radius_q, radius=args.radius) 

algo_args.env_fn = env_fn
args.env_fn = env_fn
algo_args.batch_size=128
if args.debug:
    algo_args.batch_size = 4
    algo_args.max_ep_len=2
    algo_args.replay_size=1
    algo_args.n_warmup=1
    algo_args.n_test=1
if args.test:
    algo_args.n_warmup = 0
    algo_args.n_test = 50
if args.profiling:
    algo_args.batch_size=128
    if algo_args.agent_args.p_args is None:
        algo_args.n_step = 50
    else:
        algo_args.n_step = algo_args.batch_size + 10
        algo_args.replay_size = 1000
        algo_args.n_warmup = algo_args.batch_size
    algo_args.n_test = 1
    algo_args.max_ep_len = 20
if args.seed is None:
    args.seed = np.random.randint(65536)
agent_args = algo_args.agent_args
agent_args.parallel = args.parallel
args.name = f'{args.name}_{env_fn.__name__}_{agent_args.agent.__name__}_{args.seed}'
p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args

if not p_args is None:
    print(f"rollout reuse:{(p_args.refresh_interval/q_args.update_interval*algo_args.batch_size)/algo_args.replay_size}")
# each generated data will be used so many times

import torch
torch.set_num_threads(args.n_thread)
print(f"n_threads {torch.get_num_threads()}")
print(f"n_gpus {torch.cuda.device_count()}")

ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
logger = LogServer.remote({'run_args':args, 'algo_args':algo_args}, mute=args.debug or args.test or args.profiling)
logger = LogClient(logger)
if args.profiling:
    import cProfile
    cProfile.run("RL(logger = logger, run_args=args, **algo_args._toDict()).run()",
                 filename=f'device{args.device}_parallel{args.parallel}.profile')
else:
    RL(logger = logger, run_args=args, **algo_args._toDict()).run()
