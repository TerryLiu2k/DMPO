import numpy
from models import MA2C_NC, MA2C_DIAL
class NeurCommAgent(MA2C_NC):
    def __init__(self, agent_args, seed, use_gpu):
        n_s_ls = agent_args.n_s_ls
        n_a_ls = agent_args.n_a_ls
        neighbor_mask = agent_args.neighbor_mask
        distance_mask = agent_args.distance_mask
        coop_gamma = agent_args.coop_gamma
        total_step = agent_args.total_step
        model_config = agent_args.model_config
        super().__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, total_step, model_config, seed=seed, use_gpu=use_gpu)
        self.observation_dim = agent_args.observation_dim
        self.action_space = agent_args.action_space
        self.action_filter, self.action_decoder = self._init_action_filter()
    
    def act(self, s, requires_log=False):
        ob = s
        done = False
        # pre-decision
        policy, action = self._get_policy(ob, done)
        action = self.action_filter(action)
        # post-decision
        value, n_action = self._get_value(ob, done, action)
        self._update_fingerprint(policy)
        return action

    def get_logp(self, s, a):
        policy = self.forward(s, done=False, ps=self._get_fingerprint())
        if a.ndim < s.n_dim:
            a.unsqueeze(-1)
        return numpy.take_along_axis(s, a, -1)

    def updateAgent(self, traj, clip=None):
        pass

    def add_transition(self, ob, p, action, reward, value, done):
        pass

    def save(self, info=None):
        pass

    def load(self, state_dict):
        pass

    def _reset(self):
        self._update_fingerprint(self._initial_policy())
        pass

    def _get_policy(self, ob, done, mode='train'):
        pass

    def _get_value(self, ob, done, action):
        pass

    def _update_fingerprint(self, policy):
        pass

    def _get_fingerprint(self):
        pass

    def _initial_policy(self):
        pass

    def _init_action_filter(self):
        pass


class DIALAgent(MA2C_DIAL):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, total_step, model_config, seed, use_gpu):
        super().__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, total_step, model_config, seed=seed, use_gpu=use_gpu)
    
    def act(self, s, requires_log=False):
        pass

    def get_logp(self, s, a):
        pass

    def updateAgent(self, traj, clip=None):
        pass

    def add_transition(self, ob, p, action, reward, value, done):
        pass

    def save(self, info=None):
        pass

    def load(self, state_dict):
        pass

    def _reset(self):
        pass
