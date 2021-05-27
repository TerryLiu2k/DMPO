import os
import torch


from algorithms.config.ATSC_MBPO import main
from algorithms.envs.ATSC import ATSCGrid as env_fn

"""
from algorithms.config.CACC_MBPO_factorized import main
#from algorithms.envs.CACC import CACC_slowdown as env_fn
from algorithms.envs.CACC import CACC_catchup as env_fn
"""


os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
device = 'cpu'
name = 'main'
n_cpu = 1/2
n_gpu = 0

init_checkpoint = None
print(f"num GPUs: {torch.cuda.device_count()}")
main(env_fn=env_fn, n_cpu=n_cpu, n_gpu=n_gpu, init_checkpoint=None, debug=False, test=False, seed=None, name=name, device=device)
# device ='cpu' or 'cuda', no need to set ordinal

