from copy import deepcopy
from torch.multiprocessing import Pool, Process, set_start_method
import ray
from .utils import dictSelect, dictSplit, listStack, parallelEval, locate
from .models import *
from ray.util import pdb as ppdb
import ipdb as pdb

"""
    Not implemented yet:
        PPO, SAC continous action, MBPO continous action 
    Hierarchiy:
        algorithm
            batchsize
            the number of updates per interaction
            preprocessing the env    
            Both the agent and the model do not care about the tensor shapes or model architecture
        agent
            contains models
            An agent exposes:
                .act() does the sampling
                .update_x(batch), x = p, q, pi
        (abstract) models
            q 
                takes the env, and kwargs
                when continous, = Q(s, a) 
                when discrete, 
                    Q returns Q for all/average/an action,
                    depending on the input action
            pi returns a distribution
        network
            CNN, MLP, ...
"""

class QLearning(nn.Module):
    """ Double Dueling clipped (from TD3) Q Learning"""
    def __init__(self, logger, env, q_args, gamma, eps, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__()
        self.gamma = gamma
        self.target_sync_rate=target_sync_rate
        self.eps = eps
        self.logger = logger
        self.action_space=env.action_space

        self.q1 = QCritic(env, **q_args._toDict())
        self.q2 = QCritic(env, **q_args._toDict())
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_args.lr)
        
    def _evalQ(self, s, output_distribution, a, **kwargs):
        s = s.to(self.alpha.device)
        with torch.no_grad():
            q1 = self.q1(s, output_distribution, a)
            q2 = self.q2(s, output_distribution, a)
            return torch.min(q1, q2)
        
    def updateQ(self, s, a, r, s1, d):
        
        s, a, r, s1, d = locate(self.alpha.device, s, a, r, s1, d)
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            q_next = torch.min(self.q1(s1), self.q2(s1))
            a = q_next.argmax(dim=1)
            q1_pi_targ = self.q1_target(s1, a)
            q2_pi_targ = self.q2_target(s1, a)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        self.logger.log(q=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
        self.q_optimizer.step()

        # Record things
        self.logger.log(q_update=None, q_loss=loss_q/2, rolling=100)
        
        # update the target nets
        with torch.no_grad():
            for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                for p, p_targ in zip(current.parameters(), target.parameters()):
                    p_targ.data.mul_(1 - self.target_sync_rate)
                    p_targ.data.add_(self.target_sync_rate * p.data)
                
        
    def act(self, s, deterministic=False):
        """
        o and a of shape [b, ..],
        not differentiable
        """
        s = s.to(self.alpha.device)
        with torch.no_grad():
            q1 = self.q1(s)
            q2 = self.q2(s)
            q = torch.min(q1, q2)
            a = q.argmax(dim=1)
            if not deterministic and random.random()<self.eps:
                return torch.as_tensor(self.action_space.sample())
            return a
        
    def setEps(self, eps):
        self.eps = eps
        
    def save(self, info=None):
        self.logger.save(self, info)
        
    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])

class SAC(QLearning):
    """ Actor Critic (Q function) """
    def __init__(self, logger, env, q_args, pi_args, gamma, target_entropy, target_sync_rate, alpha=0, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env, q_args, gamma, 0, target_sync_rate, **kwargs)
        
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.target_entropy = target_entropy
        
        self.eps = 0
        self.action_space = env.action_space
        if isinstance(self.action_space, Box): #continous
            self.pi = SquashedGaussianActor(**pi_args._toDict())
        else:
            self.pi = CategoricalActor(**pi_args._toDict())
                                
        if not target_entropy is None:
            pi_params = itertools.chain(self.pi.parameters(), [self.alpha])
        else:
            pi_params = self.pi.parameters()                   
        self.pi_optimizer = Adam(pi_params, lr=pi_args.lr)

    def act(self, s, deterministic=False, output_distribution=False):
        """
            o of shape [b, ..]
            not differentiable
            called during env interaction and model rollout
            not used when updating q
        """
        s = s.to(self.alpha.device)
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = self.pi(s)
                p_a = a
                # [b, n_agent, n_action] or [b, n_action]
                greedy_a = a.argmax(dim=-1)
                stochastic_a = Categorical(a).sample()
                probs = torch.ones(*a.shape)/self.action_space.n
                random_a = Categorical(probs).sample().to(s.device)
                self.logger.log(eps=self.eps)
                if  torch.isnan(a).any():
                    print('action is nan!')
                    a = random_a
                elif deterministic:
                    a = greedy_a
                elif np.random.rand()<self.eps:
                    a = random_a
                else:
                    a = stochastic_a
            else:
                a = self.pi(s, deterministic)
                if isinstance(a, tuple):
                    a = a[0]
            if output_distribution:
                return a.detach(), p_a.detach()
            else:
                return a.detach()
    
    def updatePi(self, s, q = None):
        """
        q is None for single agent
        """
        s = s.to(self.alpha.device)
        if not q is None:
            q = q.to(self.alpha.device)
        if isinstance(self.action_space, Discrete):
            pi = self.pi(s) + 1e-5 # avoid nan
            logp = torch.log(pi/pi.sum(dim=1, keepdim=True))
            if q is None:
                q1 = self.q1(s)
                q2 = self.q2(s)
                q = torch.min(q1, q2)
            q = q - self.alpha.detach() * logp
            optimum = q.max(dim=1, keepdim=True)[0].detach()
            regret = optimum - (pi*q).sum(dim=1)
            loss = regret.mean()
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            if not self.target_entropy is None:
                alpha_loss = (entropy.detach()-self.target_entropy)*self.alpha
                loss = loss + alpha_loss
            self.logger.log(pi_entropy=entropy, pi_regret=loss, alpha=self.alpha)
        else:
            action, logp = self.pi(s)
            q1 = self.q1(s, action)
            q2 = self.q2(s, action)
            q = torch.min(q1, q2)
            q = q - self.alpha.detach() * logp
            loss = (-q).mean()
            self.logger.log(logp=logp, pi_reward=q)
            
        self.pi_optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.pi.parameters(), max_norm=5, norm_type=2)
            self.pi_optimizer.step()
            if self.alpha < 0:
                self.alpha.data = torch.tensor(0, dtype=torch.float).to(self.alpha.device)

    
    def updateQ(self, s, a, r, s1, d, a1=None, p_a1=None):
        """
            uses alpha to encourage diversity
            for discrete action spaces, different from QLearning since we have a policy network
                takes all possible next actions
            a1 is determinisitc actions of neighborhood,
            only used for decentralized multiagent
            the distribution of local action is recomputed
        """
        s, a, r, s1, d, a1, p_a1 = locate(self.alpha.device, s, a, r, s1, d, a1, p_a1)
        q1 = self.q1(s, False, a)
        q2 = self.q2(s, False, a)
        
        if isinstance(self.action_space, Discrete):
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                # local a1 distribution
                loga1 = torch.log(p_a1)
                q1_pi_targ = self.q1_target(s1, True, a1) 
                q2_pi_targ = self.q2_target(s1, True, a1)  # [b, n_a]
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha.detach() * loga1
                q_pi_targ = (p_a1*q_pi_targ).sum(dim=1)
                backup = r + self.gamma * (1 - d) * (q_pi_targ)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            self.logger.log(q=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer.zero_grad()
            if not torch.isnan(loss_q).any():
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
                self.q_optimizer.step()

            # Record things
            self.logger.log(q_update=None, loss_q=loss_q/2)

            # update the target nets
            with torch.no_grad():
                for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                    for p, p_targ in zip(current.parameters(), target.parameters()):
                        p_targ.data.mul_(1 - self.target_sync_rate)
                        p_targ.data.add_(self.target_sync_rate * p.data)
        
class MBPO(SAC):
    def __init__(self, env, logger, p_args, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env, **kwargs)
        self.n_p = p_args.n_p
        if isinstance(self.action_space, Box): #continous
            ps = [None for i in range(self.n_p)]
        else:
            ps = [ParameterizedModel(env, logger,**p_args._toDict()) for i in range(self.n_p)]
        self.ps = nn.ModuleList(ps)
        self.p_params = itertools.chain(*[item.parameters() for item in self.ps])
        self.p_optimizer = Adam(self.p_params, lr=p_args.lr)
        
    def updateP(self, s, a, r, s1, d):
        s, a, r, s1, d = locate(self.alpha.device, s, a, r, s1, d)
        loss = 0
        for i in range(self.n_p):
            loss_, r_, s1_, d_ =  self.ps[i](s, a, r, s1, d)
            loss = loss + loss_
        self.p_optimizer.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.p_params, max_norm=5, norm_type=2)
        self.p_optimizer.step()
        return r_, s1_, d_
    
    def roll(self, s, a=None):
        """ batched,
            a is None as long as single agent 
            (if multiagent, set a to prevent computing .act() redundantly)
        """
        s, a = locate(self.alpha.device, s, a)
        p = self.ps[np.random.randint(self.n_p)]
        
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                if a is None:
                    a = self.act(s, deterministic=False)
                r, s1, d = p(s, a)
            else:
                return None
        return  r, s1, d

@ray.remote(num_gpus = 1/8, num_cpus=1)
class Worker(object):
    """
    A ray actor wrapper class for multiprocessing
    """
    def __init__(self, agent_fn, device, **args):
        self.gpus = ray.get_gpu_ids()
       # torch.cuda.set_per_process_memory_fraction(1/30)
        self.device = torch.device(device)
        self.instance = agent_fn(**args).to(self.device)
        
    def roll(self, **data):
        return self.instance.roll(**data)
    
    def updateP(self, **data):
        return self.instance.updateP(**data)
        
    def updateQ(self, **data):
        self.instance.updateQ(**data)
        
    def _evalQ(self, **data):
        return self.instance._evalQ(**data)
        
    def updatePi(self, **data):
        self.instance.updatePi(**data) 

    def act(self, s, deterministic=False, output_distribution=False):
        return self.instance.act(s, deterministic, output_distribution)
    
    def setEps(self, eps):
        self.instance.setEps(eps)
    
    def save(self, info=None):
        self.instance.save(info)
        
    def load(self, state_dict):
        self.instance.load(state_dict)
    
    
class MultiAgent(nn.Module):
    def __init__(self, n_agent, env, wrappers, device, **agent_args):
        """
            A meta-agent for Multi Agent RL on a factorized environment
            
            The shape of s,a,r,d
                We assume the s, a, r and d from the env are factorized for each agent, of shape [b, n_agent, ...]
                    The action_space and observation_space of the env should be for single agent
                    In general, d may be different for each agent
                
                In general, the interface may be different for each module, therefore we cannot use only one env wrapper
                    e.g. p takes k-hop s but pi takes 1-hop
                
                The pre-postprocessing, or env wrappers that broadcast, scatter and gather s, a, r, d should be registered in this class
                The wrappers are functions named: 
                    p_in(s, a) -> (s1, r, d), p_out (s, r, d)
                    q(s, a, r, s1, a1) or v(s, r, s)
                    pi_in(s, q), pi_out(a), q_out (for pi)
                    
                We make the following simplifying assumptions: 
                    the outputs of p, q, and pi are local
                    therefore, the wrappers we need are:
                        p_in(s, a) -> (s1, r, d)
                        q(s, a, r, s1, a1) or v(s, r, s), 
                        pi_in(s, q)
                    Notice p(a) during roll, pi(q) during updatePi, q(a, a1) during updateQ
                    require collecting data generated by the model,
                    and therefore these functions must non-trivially overloaded
                
                The interface shape of p, q and pi that should be explicitly configured:
                    p: the shape of s, the number of a's
                    q: the shape of s, the number of non-local a's
                        since for discrete action, only the non-local a's are embeded
                    pi: the shape of s
        """
        super().__init__()
        agent_fn = agent_args['agent']
        logger = agent_args.pop('logger')
        self.logger = logger
        self.env= env
        self.agents = []
        if not agent_args['p_args'] is None:
            self.p_to_predict = agent_args['p_args'].to_predict
        for i in range(n_agent):
            agent = Worker.remote(agent_fn=agent_fn, device=device, logger = logger.child(f"{i}"), env=env, **agent_args)
            self.agents.append(agent)
        wrappers['p_out'] = listStack
        # (s, r, d)
        wrappers['q_out'] = lambda x: torch.stack(x, dim=1)
        # (a)
        self.wrappers = wrappers

        if agent_args['p_args'] is not None:
            self.p_to_predict = agent_args['p_args'].to_predict
        # removed agent parallelism for it does not yield any performance gain
        
    def roll(self, **data):
        """
        computes the actions and reuses for each agent's model
        """
        s = data['s']
        a = self.act(s, deterministic=False)
        data['a'] = a
        data = self.wrappers['p_in'](data)
        results = parallelEval(self.agents, 'roll' ,data)
        results = self.wrappers['p_out'](results) # r, s1, d
        if not 'r' in self.p_to_predict:
            s1 = results[1]
            reward, done = self.env.state2Reward(s1)
            return reward, s1, done
        return results
    
    def updateP(self, **data):
        reward = data['r']
        data_split = self.wrappers['p_in'](data)
        results = parallelEval(self.agents, 'updateP', data_split)
        # returns r, s1, d for logging
        results = self.wrappers['p_out'](results)
        if not 'r' in self.p_to_predict:
            reward_, done_ = self.env.state2Reward(results[1])
            state_error = (results[1] - data['s1'])**2
            reward_loss = (reward - reward_)**2
            reward_var = (reward - reward.mean(dim=0, keepdim=True))**2
            self.logger.log(reward_error = reward_loss.mean(), 
                           reward_var = reward_var.mean(),
                           state_error = state_error.mean())
            debug_reward, done_ = self.env.state2Reward(data['s1'])
            debug_reward = (debug_reward - reward)**2
            self.logger.log({'debug/reward_error': debug_reward.mean()})
        
    def updateQ(self, **data):
        """
        computes a1 and reuses for each agent's model
        s, a, r, s1, d
        q(s, a), q(s1, pi(s1))
        """
        a1, p_a1 = self.act(data['s1'], deterministic=False, output_distribution=True)
        inputs = {'a1': a1,
                 'p_a1': p_a1}
        inputs.update(data)
        inputs = self.wrappers['q_in'](inputs)
        results = parallelEval(self.agents, 'updateQ', inputs)
        
    def _evalQ(self, **data):
        data = {'output_distribution': True, 
                 's': data['s'], 
                 'a': data['a']}
        data = self.wrappers['q_in'](data)
        results = parallelEval(self.agents, '_evalQ', data)
        results = self.wrappers['q_out'](results)
        return results
        
    def updatePi(self, **data):
        q = self._evalQ(**data)
        # a list of qs
        data = {'q': q,
                's': data['s'],
               }
        data = self.wrappers['pi_in'](data)
        results = parallelEval(self.agents, 'updatePi', data)    

    def act(self, s, deterministic=False, output_distribution=False):
        data = {'s': s, 'deterministic':deterministic, 
               'output_distribution': output_distribution}
        inputs = self.wrappers['pi_in'](data)
        results = parallelEval(self.agents, 'act', inputs)
        if output_distribution:
            return listStack(results)
        else:
            return torch.stack(results, dim=1)
    
    def setEps(self, eps):
        parallelEval(self.agents, 'setEps', [{'eps': eps}]*len(self.agents))
        
    def save(self, info=None):
        parallelEval(self.agents, 'save', [{'info': info}]*len(self.agents))
        ray.get(self.logger.server.save.remote(flush=True))
        
    def load(self, path):
        with open(path, "rb") as file:
            dic = torch.load(file)
        parallelEval(self.agents, 'load', [{'state_dict': dic}]*len(self.agents))
        print(f"checkpointed loaded from {path}")