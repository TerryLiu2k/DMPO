from algorithms.config.CACC_MBPO import main
from algorithms.envs.CACC import CACC_catchup, CACC_slowdown
init_checkpoint = "checkpoints/CACC_catchup_MultiagentMBPO_20905/12001_-247.26344310442602.pt"
main(env_fn=CACC_slowdown, init_checkpoint=None, debug=False, test=False, seed=None, device=0)
