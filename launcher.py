import os
import torch


from algorithms.config.ATSC_MBPO import main
from algorithms.envs.ATSC import ATSCGrid as env_fn

"""

from algorithms.config.CACC_MBPO import main
from algorithms.envs.CACC import CACC_catchup as env_fn
from algorithms.envs.CACC import CACC_slowdown as env_fn

"""


os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
init_checkpoint = "checkpoints/CACC_catchup_MultiagentMBPO_20905/12001_-247.26344310442602.pt"
print(f"num GPUs: {torch.cuda.device_count()}")
main(env_fn=env_fn, init_checkpoint=None, debug=False, test=False, seed=None, device='cpu')
# device ='cpu' or 'cuda', no need to set ordinal
