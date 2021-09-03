from logging import log
import os
from re import T
import ray
import time
import warnings
from algorithms.utils import Config, LogClient, LogServer
from algorithms.envs.Flow import makeFlowGrid, makeFlowGridTest, makeVectorizedFlowGridFn
from algorithms.config.FLOW_PPO import getArgs
from algorithms.mbdppo.MB_DPPO import OnPolicyRunner
from algorithms.mbdppo.MB_DPPO import DPPOAgent as agent_fn
import torch

warnings.filterwarnings('ignore')

def getEnvArgs():
    env_args = Config()
    env_args.n_env = 2
    env_args.n_cpu = 1 # per environment
    env_args.n_gpu = 0
    return env_args

def getRunArgs():
    run_args = Config()
    run_args.n_thread = 1
    run_args.parallel = False
    run_args.device = 'cpu'
    run_args.n_cpu = 1/4
    run_args.n_gpu = 0
    run_args.debug = True
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

def initArgs(run_args, env_train, env_test):
    ref_env = env_train
    alg_args = getArgs(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    return alg_args

def initAgent(logger, device, agent_args):
    return agent_fn(logger, device, agent_args)

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
        alg_args.n_warmup=1
        alg_args.n_test=1
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

env_args = getEnvArgs()
env_fn_train = makeVectorizedFlowGridFn(env_args)
env_fn_test = makeFlowGridTest
env_train = env_fn_train()
env_test = env_fn_test()
run_args = getRunArgs()
alg_args = initArgs(run_args, env_train, env_test)
alg_args, run_args = override(alg_args, run_args, env_fn_train)

os.environ['CUDA_VISIBLE_DEVICES']='0'
ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
logger = LogServer.remote({'run_args':run_args, 'algo_args':alg_args}, mute=run_args.debug or run_args.test or run_args.profiling)
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
