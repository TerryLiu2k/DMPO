import itertools
from sys import setdlopenflags
from numpy.core.numeric import zeros_like
#from torch.autograd.grad_mode import F
from torch.utils.data import dataloader
from tqdm.std import trange
from algorithms.algorithm import ReplayBuffer
from ray.state import actors
from algorithms.utils import collect
from gym.spaces.discrete import Discrete
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.utils.data as Data
from torch.optim import Adam, optimizer
import numpy as np
import pickle

from algorithms.models import CategoricalActor, SquashedGaussianActor

class MultiCollect:
    def __init__(self, adjacency, device="cpu"):
        """
        Method: 'gather', 'reduce_mean', 'reduce_sum'.
        Adjacency: torch Tensor.
        Everything outward would be in the same device specifed in the initialization parameter.
        """
        self.device = device
        n = adjacency.size()[0]
        adjacency = adjacency > 0 # Adjacency Matrix, with size n_agent*n_agent. 
        adjacency = adjacency | torch.eye(n, device=device).bool() # Should contain self-loop, because an agent should utilize its own info.
        adjacency = adjacency.to(device)
        self.degree = adjacency.sum(dim=1) # Number of information available to the agent.
        self.indices = []
        index_full = torch.arange(n, device=device)
        for i in range(n):
            self.indices.append(torch.masked_select(index_full, adjacency[i])) # Which agents are needed.

    def gather(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim_i] for i in range(n_agent)]
        """
        return self._collect('gather', tensor)

    def reduce_mean(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_mean', tensor)

    def reduce_sum(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_sum', tensor)

    def _collect(self, method, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: 
            gather: [[batch_size, dim_i] for i in range(n_agent)]
            reduce: [batch_size, n_agent, dim]  # same as input
        """
        tensor = tensor.to(self.device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        result = []
        for i in range(n):
            if method == 'gather':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).view(b, -1))
            elif method == 'reduce_mean':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).mean(dim=1))
            else:
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).sum(dim=1))
        if method != 'gather':
            result = torch.stack(result, dim=1)
        return result


class TrajectoryBuffer:
    def __init__(self, device="cpu"):
        self.device = device
        self.s, self.a, self.r, self.s1, self.d, self.logp = [], [], [], [], [], []
    
    def store(self, s, a, r, s1, d, logp):
        """
        Would be converted into [batch_size, n_agent, dim].
        """
        device = self.device
        [s, r, s1, logp] = [torch.as_tensor(item, device=device, dtype=torch.float) for item in [s, r, s1, logp]]
        d = torch.as_tensor(d, device=device, dtype=torch.bool)
        a = torch.as_tensor(a, device=device)
        while s.dim() <= 2:
            s = s.unsqueeze(dim=0)
        b, n, dim = s.size()
        if d.dim() <= 1:
            d = d.unsqueeze(0)
        d = d[:, :n]
        if r.dim() <= 1:
            r = r.unsqueeze(0)
        r = r[:, :n]
        [s, a, r, s1, d, logp] = [item.view(b, n, -1) for item in [s, a, r, s1, d, logp]]
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.d.append(d)
        self.logp.append(logp)
    
    def retrieve(self):
        """
        Returns a dictionary with s, a, r, s1, d, logp.
        Data are of size [n_trajectory, T, n_agent, dim]
        """
        names = ["s", "a", "r", "s1", "d", "logp"]
        traj_all = {}
        for name in names:
            traj_all[name] = torch.stack(self.__getattribute__(name), dim=1)
        return traj_all
        


class OnPolicyRunner:
    def __init__(self, logger, run_args, alg_args, agent, env_learn, env_test, **kwargs):
        self.logger = logger
        self.name = run_args.name
        if not run_args.init_checkpoint is None:
            agent.load(run_args.init_checkpoint)
            logger.log(interaction=run_args.start_step)  
        self.start_step = run_args.start_step 

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_warmup = alg_args.n_warmup
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.test_length = alg_args.test_length
        self.max_episode_len = alg_args.max_episode_len
        self.clip_scheme = None if (not hasattr(alg_args, "clip_scheme")) else alg_args.clip_scheme
        
        # agent initialization
        self.agent = agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        # environment initialization
        self.env_learn = env_learn
        self.env_test = env_test

        # buffer initialization
        self.discrete = agent.discrete
        action_dtype = torch.long if self.discrete else torch.float
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.model_buffer = ReplayBuffer(max_size=alg_args.model_buffer_size, action_dtype=action_dtype)
        self.s, self.episode_len, self.episode_reward = self.env_learn.reset(), 0, 0

    def run(self):

        if self.n_warmup > 0 and self.model_based:
            self.rollout_env()
            for _ in range(self.n_model_update_warmup):
                batch = self.model_buffer.sampleBatch(self.model_batch_size)
                self.agent.updateModel(**batch)
            
        for iter in trange(self.n_iter):
            if iter % self.test_interval == 0:
                self.test()
            if self.model_based:
                for _ in range(self.n_model_update):
                    batch = self.model_buffer.sampleBatch(self.model_batch_size)
                    self.agent.updateModel(**batch)
            traj = self.rollout_env()
            if self.model_based:
                traj = self.rollout_model(traj)
            if self.clip_scheme is not None:
                self.agent.updateAgent(traj, self.clip_scheme(iter))
            else:
                self.agent.updateAgent(traj)
            del traj

    def test(self):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        length = self.test_length
        returns = []
        scaled = []
        lengths = []
        episodes = []
        for i in trange(self.n_test):
            episode = []
            env = self.env_test
            env.reset()
            d, ep_ret, ep_len = np.array([False]), 0, 0
            while not(d.any() or (ep_len == length)):
                s = env.get_state_()
                s = torch.as_tensor(s, dtype=torch.float, device=self.device)
                a = self.agent.act(s, requires_log=False) # a and logp are Tensors
                a = a.squeeze(0).detach().cpu().numpy()
                s1, r, d, _ = env.step(a)
                episode += [(s.tolist(), a.tolist(), r.tolist())]
                d = np.array(d)
                ep_ret += r.sum()
                ep_len += 1
                self.logger.log(interaction=None)
            if hasattr(env, 'rescaleReward'):
                scaled += [ep_ret]
                ep_ret = env.rescaleReward(ep_ret, ep_len)
            returns += [ep_ret]
            lengths += [ep_len]
            episodes += [episode]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns, test_episode_len=lengths, test_round=None)
        print(returns)
        print(f"{self.n_test} episodes average accumulated reward: {returns.mean()}")
        if hasattr(env, 'rescaleReward'):
            print(f"scaled reward {np.mean(scaled)}")
        with open(f"checkpoints/{self.name}/test.pickle", "wb") as f:
            pickle.dump(episodes, f)
        with open(f"checkpoints/{self.name}/test.txt", "w") as f:
            for episode in episodes:
                for step in episode:
                    f.write(f"{step[0]}, {step[1]}, {step[2]}\n")
                f.write("\n")
        return returns.mean()

    def rollout_env(self, length = 0):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        if length <= 0:
            length = self.rollout_length
        env = self.env_learn
        traj = TrajectoryBuffer(device=self.device)
        for t in range(length):
            s = env.get_state_()
            s = torch.as_tensor(s, dtype=torch.float, device=self.device)
            a, logp = self.agent.act(s, requires_log=True) # a and logp are Tensors
            a = a.squeeze(0).detach().cpu().numpy()
            s1, r, d, _ = env.step(a)
            self.episode_reward += r
            self.episode_len += 1
            self.logger.log(interaction=None)
            if self.episode_len == self.max_episode_len:
                d = np.zeros(d.shape, dtype=np.float32)
            d = np.array(d)
            if self.model_based:
                self.model_buffer.store(s, a, r, s1, d)
            traj.store(s, a, r, s1, d, logp)
            if d.any() or (self.episode_len == self.max_episode_len):
                self.logger.log(episode_reward=self.episode_reward.sum(), episode_len = self.episode_len, episode=None)
                _, self.episode_reward, self.episode_len = self.env_learn.reset(), 0, 0
        return traj.retrieve()
    
    def rollout_model(self, traj):
        pass



class DPPOAgent(nn.ModuleList):
    """
    Everything in and out is torch Tensor.
    """
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__()
        self.logger = logger
        self.device = device
        self.n_agent = agent_args.n_agent
        self.gamma = agent_args.gamma
        self.lamda = agent_args.lamda
        self.clip = agent_args.clip
        self.target_kl = agent_args.target_kl
        self.v_coeff = agent_args.v_coeff
        self.entropy_coeff = agent_args.entropy_coeff
        self.lr = agent_args.lr
        self.n_update_v = agent_args.n_update_v
        self.n_update_pi = agent_args.n_update_pi
        self.n_minibatch = agent_args.n_minibatch
        self.use_reduced_v = agent_args.use_reduced_v

        self.advantage_norm = agent_args.advantage_norm

        self.observation_space = agent_args.observation_space
        self.observation_dim = self.observation_space.shape[0]
        self.action_space = agent_args.action_space
        self.discrete = isinstance(agent_args.action_space, Discrete)
        if self.discrete:
            self.action_dim = self.action_space.n
            self.action_shape = self.action_dim
        else:
            self.action_shape = self.action_space.shape
            self.action_dim = 1
            for j in self.action_shape:
                self.action_dim *= j
        
        self.adj = torch.as_tensor(agent_args.adj, device=self.device, dtype=torch.float)
        self.radius_v = agent_args.radius_v
        self.radius_pi = agent_args.radius_pi
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        self.collect_pi, self.actors = self._init_actors()
        self.collect_v, self.vs = self._init_vs()

        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr)
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)

    def act(self, s, requires_log=False):
        """
        Discrete only.
        This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
        """
        with torch.no_grad():
            while s.dim() <= 2:
                s = s.unsqueeze(0)
            s = s.to(self.device)
            s = self.collect_pi.gather(s)
            actions = []
            log_probs = []
            for i in range(self.n_agent):
                probs = self.actors[i](s[i])
                distrib = Categorical(probs)
                action = distrib.sample()
                log_prob = distrib.log_prob(action)
                actions.append(action.detach())
                log_probs.append(log_prob.detach())
            if requires_log:
                return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)
            else:
                return torch.stack(actions, dim=1)
    
    def get_logp(self, s, a):
        s = s.to(self.device)
        s = self.collect_pi.gather(s)
        log_prob = []
        for i in range(self.n_agent):
            probs = self.actors[i](s[i])
            log_prob.append(torch.log(torch.gather(probs, dim=-1, index=torch.select(a, dim=1, index=i).long())))
        return torch.stack(log_prob, dim=1)

    def updateAgent(self, traj, clip=None):        
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch
        s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        value_old, returns, advantages, reduced_advantages = self._process_traj(traj)

        advantages_old = reduced_advantages if self.use_reduced_v else advantages

        b, T, n, d_s = s.size()
        d_a = a.size()[-1]
        s = s.view(-1, n, d_s)
        a = a.view(-1, n, d_a)
        logp = logp.view(-1, n, 1)
        advantages_old = advantages_old.view(-1, n, 1)
        returns = returns.view(-1, n, 1)
        value_old = value_old.view(-1, n, 1)

        batch_total = logp.size()[0]
        batch_size = int(batch_total/n_minibatch)
        kl = 0
        i_pi = 0
        for i_pi in range(self.n_update_pi):
            batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
            if n_minibatch > 1:
                idxs = np.random.randint(0, len(batch_total), size=batch_size)
                [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in [batch_state, batch_action, batch_logp, batch_advantages_old]]
            batch_logp_new = self.get_logp(batch_state, batch_action)
            logp_diff = batch_logp_new - batch_logp
            kl = logp_diff.mean()
            ratio = torch.exp(batch_logp_new - batch_logp)
            surr1 = ratio * batch_advantages_old
            surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
            loss_surr = torch.min(surr1, surr2).mean()
            loss_entropy = - torch.mean(torch.exp(batch_logp_new) * batch_logp_new)
            loss_pi = - loss_surr - self.entropy_coeff * loss_entropy
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()
            self.logger.log(surr_loss = loss_surr, entropy = loss_entropy, kl_divergence = kl, pi_update=None)
            if self.target_kl is not None and kl > 1.5 * self.target_kl:
                break
        self.logger.log(pi_update_step=i_pi)

        for _ in range(self.n_update_v):
            batch_returns = returns
            batch_state = s
            if n_minibatch > 1:
                idxs = np.random.randint(0, len(batch_total), size=batch_size)
                [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
            batch_v_new = self._evalV(batch_state)
            loss_v = ((batch_v_new - batch_returns) ** 2).mean()
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()
            self.logger.log(v_loss=loss_v, v_update=None)
        self.logger.log(update=None, reward=r, value=value_old, clip=clip, returns=returns, advantages=advantages_old)
        

    def save(self, info=None):
        self.logger.save(self, info)

    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])

    def _evalV(self, s):
        s = s.to(self.device)
        s = self.collect_v.gather(s)
        values = []
        for i in range(self.n_agent):
            values.append(self.vs[i](s[i]))
        return torch.stack(values, dim=1)

    def _init_actors(self):
        collect_pi = MultiCollect(torch.matrix_power(self.adj, self.radius_pi), device=self.device)
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = collect_pi.degree[i] * self.observation_dim
            if self.discrete:
                actors.append(CategoricalActor(**self.pi_args._toDict()))
            else:
                actors.append(SquashedGaussianActor(**self.pi_args._toDict()))
        return collect_pi, actors
    
    def _init_vs(self):
        collect_v = MultiCollect(torch.matrix_power(self.adj, self.radius_v), device=self.device)
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.v_args.sizes[0] = collect_v.degree[i] * self.observation_dim
            v_fn = self.v_args.network
            vs.append(v_fn(**self.v_args._toDict()))
        return collect_v, vs
    
    def _process_traj(self, traj):
        with torch.no_grad():
            b, T, n, dim_s = traj['s'].size()
            s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            value = self._evalV(s.view(-1, n, dim_s)).view(b, T, n, -1)

            returns = torch.zeros(value.size(), device=self.device)
            deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)
            prev_value = self._evalV(s1.select(1, T - 1))
            prev_return = torch.zeros_like(prev_value)
            prev_advantage = torch.zeros_like(prev_return)
            d_mask = d.float()
            for t in reversed(range(T)):
                returns[:, t, :, :] = r.select(1, t) + self.gamma * (1-d_mask.select(1,t)) * prev_return
                deltas[:, t, :, :]= r.select(1, t) + self.gamma * (1-d_mask.select(1,t)) * prev_value - value.select(1, t).detach()
                advantages[:, t, :, :] = deltas.select(1, t) + self.gamma * self.lamda * (1-d_mask.select(1,t)) * prev_advantage

                prev_return = returns.select(1, t)
                prev_value = value.select(1, t)
                prev_advantage = advantages.select(1, t)
            reduced_advantages = self.collect_v.reduce_sum(advantages.view(-1, n, 1)).view(advantages.size())
            if self.advantage_norm:
                reduced_advantages = (reduced_advantages - reduced_advantages.mean(dim=1, keepdim=True)) / (reduced_advantages.std(dim=1, keepdim=True) + 1e-5)
                advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-5)
        return value, returns, advantages, reduced_advantages
        

class MB_DPPOAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__(logger, device, agent_args)
        self.radius_p = agent_args.radius_p
        self.p_args = agent_args.p_args
        self.collect_p, self.ps = self._init_ps()
    
    def updateAgent(self):
        pass

    def updateModel(self, **kwargs):
        pass

    def _rollout(self):
        pass

    def _init_ps(self):
        pass
    
    def _rollout_model(self, traj, length = 0, n_instance = 0):
        pass
