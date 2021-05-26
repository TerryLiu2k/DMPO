import os
from algorithms.config.CACC_MBPO import main
from algorithms.config.ATSC_MBPO import main
from algorithms.envs.CACC import CACC_catchup, CACC_slowdown
from algorithms.envs.ATSC import ATSCGrid

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
init_checkpoint = "checkpoints/CACC_catchup_MultiagentMBPO_20905/12001_-247.26344310442602.pt"
main(env_fn=ATSCGrid, init_checkpoint=None, debug=False, test=False, seed=None)
