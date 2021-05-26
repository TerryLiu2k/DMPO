import os

import torch

from algorithms.config.CACC_MBPO_conservative import main
#from algorithms.envs.CACC import CACC_slowdown as env_fn
from algorithms.envs.CACC import CACC_catchup as env_fn

<<<<<<< HEAD
"""
=======


>>>>>>> 1069b6c99b09cbdc150a64e2a4284315a4493d3e
from algorithms.config.ATSC_MBPO import main
from algorithms.envs.ATSC import ATSCGrid as env_fn
"""
<<<<<<< HEAD
=======
from algorithms.config.CACC_MBPO_conservative import main
#from algorithms.envs.CACC import CACC_catchup as env_fn
from algorithms.envs.CACC import CACC_slowdown as env_fn
"""
>>>>>>> 1069b6c99b09cbdc150a64e2a4284315a4493d3e


os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
<<<<<<< HEAD
device = 'cuda'
name = 'small_tau'
=======
device = 'cpu'
name = 'main'
>>>>>>> 1069b6c99b09cbdc150a64e2a4284315a4493d3e
init_checkpoint = None
print(f"num GPUs: {torch.cuda.device_count()}")
main(env_fn=env_fn, init_checkpoint=None, debug=False, test=False, seed=None, name=name, device=device)
# device ='cpu' or 'cuda', no need to set ordinal
