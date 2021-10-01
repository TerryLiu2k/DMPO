from logging import log
import os
from re import T
import ray
import time
import warnings
from algorithms.utils import Config, LogClient, LogServer
from algorithms.envs.Flow import makeFlowGrid, makeFlowGridTest, makeVectorizedFlowGridFn
from algorithms.envs.FigureEight import makeFigureEight2, makeFigureEightTest
from algorithms.envs.Ring import makeRingAttenuation
from algorithms.envs.CACC import CACC_catchup, CACC_slowdown
from algorithms.config.Eight_IA2C import getArgs as getArgs_eight_IA2C
from algorithms.config.Eight_IA2C import getArgs as getArgs_ring_IA2C
from algorithms.config.Catchup_IA2C import getArgs as getArgs_catchup_IA2C
from algorithms.config.Slowdown_IA2C import getArgs as getArgs_slowdown_IA2C

from algorithms.mbdppo.MB_DPPO import OnPolicyRunner
from algorithms.mbdppo.MB_DPPO import IA2C as agent_fn
from algorithms.mbdppo.MB_DPPO import DPPOAgent
from

import torch
import argparse

warnings.filterwarnings('ignore')

def getEnvArgs():
    env_args = Config()
    env_args.n_env = 10
    env_args.n_cpu = 10 # per environment
    env_args.n_gpu = 0
    return env_args

def getRunArgs():
    run_args = Config()
    run_args.n_thread = 10
    run_args.parallel = False
    run_args.device = 'cpu'
    run_args.n_cpu = 1/4
    run_args.n_gpu = 0
    run_args.debug = False
    run_args.test = False
    run_args.profiling = False
    run_args.name = 'standard'
    run_args.radius_v = 3
    run_args.radius_pi = 1
    run_args.radius_p = 1
    run_args.init_checkpoint = None
    run_args.start_step = 0
    run_args.save_period = 1800 # in seconds
    run_args.log_period = int(20)
    run_args.seed = None
    return run_args

def initArgs(run_args, env_train, env_test, input_arg):
    ref_env = env_train
    # TODO: should also consider algo
    if input_arg.env == 'eight':
        alg_args = getArgs_eight_IA2C(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    elif input_arg.env == 'ring':
        alg_args = getArgs_ring_IA2C(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    elif input_arg.env == 'catchup':
        run_args.radius_v = 2
        run_args.radius_pi = 1
        run_args.radius_p = 1
        alg_args = getArgs_catchup_IA2C(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    elif input_arg.env == 'slowdown':
        run_args.radius_v = 2
        run_args.radius_pi = 1
        run_args.radius_p = 1
        alg_args = getArgs_slowdown_IA2C(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    else:
        alg_args = None
    return alg_args

def initAgent(logger, device, agent_args):
    return agent_fn(logger, device, agent_args)

def initEnv(input_args):
    if input_args.env == 'eight':
        env_fn_train, env_fn_test = makeFigureEight2, makeFigureEightTest
    elif input_args.env == 'ring':
        env_fn_train, env_fn_test = makeRingAttenuation, makeRingAttenuation
    elif input_args.env == 'catchup':
        env_fn_train, env_fn_test = CACC_catchup, CACC_catchup
    elif input_args.env == 'slowdown':
        env_fn_train, env_fn_test = CACC_slowdown, CACC_slowdown
    else:
        env_fn_train, env_fn_test = None
    return env_fn_train, env_fn_test

def override(alg_args, run_args, env_fn_train):
    alg_args.env_fn = env_fn_train
    agent_args = alg_args.agent_args
    p_args, v_args, pi_args = agent_args.p_args, agent_args.v_args, agent_args.pi_args
    if run_args.debug:
        alg_args.model_batch_size = 4
        alg_args.max_ep_len=5
        alg_args.rollout_length = 5
        alg_args.test_length = 1
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 10
    if run_args.test:
        alg_args.n_warmup = 0
        alg_args.n_test = 10
    if run_args.profiling:
        alg_args.model_batch_size = 128
        alg_args.n_warmup = 0
        if alg_args.agent_args.p_args is None:
            alg_args.n_iter = 10
        else:
            alg_args.n_iter = 10
            alg_args.model_buffer_size = 1000
            alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.max_ep_len = 400
        alg_args.rollout_length = 400
        alg_args.test_length = 1
        alg_args.test_interval = 100
    if run_args.seed is None:
        run_args.seed = int(time.time()*1000)%65536
    agent_args.parallel = run_args.parallel
    run_args.name = '{}_{}_{}_{}'.format(run_args.name, env_fn_train.__name__, agent_fn.__name__, run_args.seed)
    return alg_args, run_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=False, default='catchup', help="environment(eight/ring/catchup/slowdown)")
    parser.add_argument('--algo', type=str, required=False, default='IA2C', help="algorithm(DMPPO/IA2C/IC3NET) ")
    args = parser.parse_args()
    '''
    if not args.option:
        parser.print_help()
        exit(1)
    '''
    return args


# get arg from cli
input_args = parse_args()

env_args = getEnvArgs()
env_fn_train, env_fn_test = initEnv(input_args)
env_train = env_fn_train()
env_test = env_fn_test()
run_args = getRunArgs()
alg_args = initArgs(run_args, env_train, env_test, input_args)
alg_args, run_args = override(alg_args, run_args, env_fn_train)

os.environ['CUDA_VISIBLE_DEVICES']='0'
#ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
logger = LogServer({'run_args':run_args, 'algo_args':alg_args}, mute=run_args.debug or run_args.test or run_args.profiling)
logger = LogClient(logger)
agent = initAgent(logger, run_args.device, alg_args.agent_args)

torch.set_num_threads(run_args.n_thread)
print(f"n_threads {torch.get_num_threads()}")
print(f"n_gpus {torch.cuda.device_count()}")

if run_args.profiling:
    import cProfile
    cProfile.run("OnPolicyRunner(logger = logger, run_args=run_args, alg_args=alg_args, agent=agent, env_learn=env_train, env_test = env_test).run()",
                 filename=f'device{run_args.device}_parallel{run_args.parallel}.profile')
else:
    OnPolicyRunner(logger = logger, run_args=run_args, alg_args=alg_args, agent=agent, env_learn=env_train, env_test = env_test).run()
