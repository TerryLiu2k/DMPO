from copy import deepcopy
from torch.multiprocessing import Pool, Process, set_start_method

from .utils import dictSelect, dictSplit, listStack, parallelEval
from .models import *

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
    def __init__(self, logger, env_fn, q_args, gamma, eps, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__()
        self.logger = logger.child("QLearningAgent")
        self.gamma = gamma
        self.target_sync_rate=target_sync_rate
        self.eps = eps
        self.action_space=env_fn().action_space

        self.q1 = QCritic(**q_args._toDict())
        self.q2 = QCritic(**q_args._toDict())
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_args.lr)
        
    def _evalQ(self, s, output_distribution, a, **kwargs):
        with torch.no_grad():
            q1 = self.q1(s, output_distribution, a)
            q2 = self.q2(s, output_distribution, a)
            return torch.min(q1, q2)
        
    def updateQ(self, s, a, r, s1, d):
        
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
    
class SAC(QLearning):
    """ Actor Critic (Q function) """
    def __init__(self, logger, env_fn, q_args, pi_args, gamma, target_entropy, target_sync_rate, alpha=0, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env_fn, q_args, gamma, 0, target_sync_rate, **kwargs)
        self.logger = logger.child("SAC")
        
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.target_entropy = target_entropy
        
        self.eps = 1 # linearly decrease to 0,
        #switching from 1 to 0 in a sudden causes nan on some tasks
        self.action_space = env_fn().action_space
        if isinstance(self.action_space, Box): #continous
            self.pi = SquashedGaussianActor(**pi_args._toDict())
        else:
            self.pi = CategoricalActor(**pi_args._toDict())
                                
        if not target_entropy is None:
            pi_params = itertools.chain(self.pi.parameters(), [self.alpha])
        else:
            pi_params = self.pi.parameters()                   
        self.pi_optimizer = Adam(pi_params, lr=pi_args.lr)

    def act(self, s, deterministic=False):
        """
            o of shape [b, ..]
            not differentiable
            called during env interaction and model rollout
            not used when updating q
        """
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = self.pi(s)
                # [b, n_agent, n_action] or [b, n_action]
                greedy_a = a.argmax(dim=-1)
                stochastic_a = Categorical(a).sample()
                probs = torch.ones(*a.shape)/self.action_space.n
                random_a = Categorical(probs).sample().to(s.device)
                self.logger.log(eps=self.eps)
                if  torch.isnan(a).any():
                    print('action is nan!')
                    return random_a
                elif deterministic:
                    return greedy_a
                elif np.random.rand()<self.eps:
                    return random_a
                return stochastic_a
            else:
                a = self.pi(s, deterministic)
                if isinstance(a, tuple):
                    a = a[0]
            return a.detach()
    
    def updatePi(self, s, q = None):
        """
        q is None for single agent
        """
        if isinstance(self.action_space, Discrete):
            pi = self.pi(s) + 1e-5 # avoid nan
            logp = torch.log(pi/pi.sum(dim=1, keepdim=True))
            if q is None:
                q1 = self.q1(s)
                q2 = self.q2(s)
                q = torch.min(q1, q2)
            q = q - self.alpha * logp
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
            q = q - self.alpha * logp
            loss = (-q).mean()
            self.logger.log(logp=logp, pi_reward=q)
            
        self.pi_optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.pi.parameters(), max_norm=5, norm_type=2)
            self.pi_optimizer.step()

    
    def updateQ(self, s, a, r, s1, d, a1=None):
        """
            uses alpha to encourage diversity
            for discrete action spaces, different from QLearning since we have a policy network
                takes all possible next actions
            a1 is determinisitc actions of neighborhood,
            only used for decentralized multiagent
            the distribution of local action is recomputed
        """
        q1 = self.q1(s, False, a)
        q2 = self.q2(s, False, a)
        
        if isinstance(self.action_space, Discrete):
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                p_a1 = self.pi(s1).detach() # [b, n_a]
                # local a1 distribution
                loga1 = torch.log(p_a1)
                q1_pi_targ = self.q1_target(s1, True, a1) 
                q2_pi_targ = self.q2_target(s1, True, a1)  # [b, n_a]
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * loga1
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
    def __init__(self, env_fn, logger, p_args, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env_fn, **kwargs)
        logger = logger.child("MBPO")
        self.logger = logger
        self.n_p = p_args.n_p
        if isinstance(self.action_space, Box): #continous
            ps = [None for i in range(self.n_p)]
        else:
            ps = [ParameterizedModel(env_fn, logger,**p_args._toDict()) for i in range(self.n_p)]
        self.ps = nn.ModuleList(ps)
        self.p_params = itertools.chain(*[item.parameters() for item in self.ps])
        self.p_optimizer = Adam(self.p_params, lr=p_args.lr)
        
    def updateP(self, s, a, r, s1, d):
        loss = 0
        for i in range(self.n_p):
            loss, r_, s1_, d_ =  self.ps[i](s, a, r, s1, d)
            loss = loss + loss
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
        p = self.ps[np.random.randint(self.n_p)]
        
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                if a is None:
                    a = self.act(s, deterministic=False)
                r, s1, d = p(s, a)
            else:
                return None
        return  r, s1, d
    
class MultiAgent(nn.Module):
    def __init__(self, n_agent, env_fn, wrappers, **agent_args):
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
        self.env= env_fn()
        self.env.reset()
        self.agents = []
        for i in range(n_agent):
            self.agents.append(agent_fn(logger = logger.child(f"{i}"), env_fn=env_fn, **agent_args))
        self.agents = nn.ModuleList(self.agents)
        wrappers['p_out'] = listStack
        # (s, r, d)
        wrappers['pi_out'] = lambda x: torch.stack(x, dim=1)
        wrappers['q_out'] = lambda x: torch.stack(x, dim=1)
        # (a)
        self.wrappers = wrappers
        for attr in ['ps', 'q1', 'pi']:
            if hasattr(self.agents[0], attr):
                setattr(self, attr, True)
        self.pool = None #Pool(n_agent)
        # removed agent parallelism for it does not yield any performance gain
        
    def roll(self, **data):
        """
        computes the actions and reuses for each agent's model
        """
        s = data['s']
        a = self.act(s, deterministic=False)
        data['a'] = a
        data['func'] = 'roll'
        data['agent'] = self.agents
        data = self.wrappers['p_in'](data)
        results = parallelEval(self.pool, data)
        results = self.wrappers['p_out'](results) # r, s1, d
        if hasattr(self.env, 'state2Reward'):
            s1 = results[1]
            reward, done = self.env.state2Reward(s1)
            return reward, s1, done
        return results
    
    def updateP(self, **data):
        data['func'] = 'updateP'
        data['agent'] = self.agents
        reward = data['r']
        data_split = self.wrappers['p_in'](data)
        results = parallelEval(self.pool, data_split)
        results = self.wrappers['p_out'](results)
        if hasattr(self.env, 'state2Reward'):
            reward_, done_ = self.env.state2Reward(results[0])
            reward_loss = (reward - reward_)**2
            reward_var = (reward - reward.mean(dim=0, keepdim=True))**2
            self.logger.log(reward_error = reward_loss.mean(), 
                           reward_var = reward_var.mean())
        
    def updateQ(self, **data):
        """
        computes a1 and reuses for each agent's model
        """
        data['func'] = 'updateQ'
        data['agent'] = self.agents
        data['a1'] = self.act(data['s1'], deterministic=False)
        data = self.wrappers['q_in'](data)
        results = parallelEval(self.pool, data)
        
    def _evalQ(self, **data):
        data = {'output_distribution': True, 
                 'agent': self.agents, 
                 's': data['s'], 
                 'a': data['a'],
                 'func': '_evalQ'}
        data = self.wrappers['q_in'](data)
        results = parallelEval(self.pool, data)
        results = self.wrappers['q_out'](results)
        return results
        
    def updatePi(self, **data):
        q = self._evalQ(**data)
        # a list of qs
        data = {'q': q,
               'func': 'updatePi',
                's': data['s'],
                'agent': self.agents
               }
        data = self.wrappers['pi_in'](data)
        results = parallelEval(self.pool, data)    

    def act(self, S, deterministic=False):
        data = {'s': S, 'deterministic':deterministic, 'func': 'act', 'agent': self.agents}
        data = self.wrappers['pi_in'](data)
        results = parallelEval(self.pool, data)
        return self.wrappers['pi_out'](results)
    
    def setEps(self, eps):
        for agent in self.agents:
            agent.setEps(eps)