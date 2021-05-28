import os
import numpy as np
import ray
import torch
from algorithms.utils import Config, LogClient, LogServer
from algorithms.algorithm import RL

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'

"""
This section contains run args, separated from args for the RL algorithm and agents
"""
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
#from algorithms.envs.CACC import CACC_catchup as env_fn
from algorithms.config.CACC_MBPO_conservative import getArgs
from algorithms.envs.CACC import CACC_slowdown as env_fn

args = Config()
args.save_period=1800 # in seconds
args.log_period=int(20)
args.seed = None
args.init_checkpoint = None
args.start_step = 0
args.debug = False
args.test = False # if no training, only test
args.profiling = True
args.parallel = False
args.name = 'test'
args.device = 'cpu'
args.n_cpu = 1 # per agent
args.n_gpu = 0
args.radius_q=2
args.radius=2

algo_args = getArgs(radius_q=args.radius_q, radius=args.radius) 

algo_args.env_fn = env_fn
args.env_fn = env_fn
algo_args.batch_size=256
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
    algo_args.batch_size=256
    algo_args.n_step = algo_args.batch_size + 50
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

ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
logger = LogServer.remote({'run_args':args, 'algo_args':algo_args}, mute=args.debug or args.test)
logger = LogClient(logger)
if args.profiling:
    import cProfile
    cProfile.run("RL(logger = logger, run_args=args, **algo_args._toDict()).run()", filename='cpu_parallel.profile')
else:
    RL(logger = logger, run_args=args, **algo_args._toDict()).run()
