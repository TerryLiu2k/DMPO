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
        self.logger = logger
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
        self.logger.log(q_update=None, loss_q=loss_q/2, rolling=100)
        
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
    def __init__(self, logger, env_fn, q_args, pi_args, alpha, gamma, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env_fn, q_args, gamma, 0, target_sync_rate, **kwargs)
        self.alpha = alpha
        self.eps = 1 # linearly decrease to 0,
        #switching from 1 to 0 in a sudden causes nan on some tasks
        self.action_space = env_fn().action_space
        if isinstance(self.action_space, Box): #continous
            self.pi = SquashedGaussianActor(**pi_args._toDict())
        else:
            self.pi = CategoricalActor(**pi_args._toDict())
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_args.lr)

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
    
    def updatePi(self, s, a, r, s1, d):
        if isinstance(self.action_space, Discrete):
            pi = self.pi(s) + 1e-5 # avoid nan
            logp = torch.log(pi/pi.sum(dim=1, keepdim=True))
            q1 = self.q1(s)
            q2 = self.q2(s)
            q = torch.min(q1, q2)
            q = q - self.alpha * logp
            optimum = q.max(dim=1, keepdim=True)[0].detach()
            regret = optimum - (pi*q).sum(dim=1)
            loss = regret.mean()
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            self.logger.log(entropy=entropy, pi_regret=loss)
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

    
    def updateQ(self, s, a, r, s1, d):
        """
            uses alpha to encourage diversity
            for discrete action spaces, different from QLearning since we have a policy network
                takes all possible next actions
        """
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        
        if isinstance(self.action_space, Discrete):
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                q_next = torch.min(self.q1(s1), self.q2(s1))
                a2 = self.pi(s1).detach() # [b, n_a]
                loga2 = torch.log(a2)
                q1_pi_targ = self.q1_target(s1)
                q2_pi_targ = self.q2_target(s1) # [b, n_a]
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * loga2
                q_pi_targ = (a2*q_pi_targ).sum(dim=1)
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
    def __init__(self, logger, env_fn, p_args, q_args, pi_args, alpha, gamma, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env_fn, q_args, pi_args, alpha, gamma, target_sync_rate, **kwargs)
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
            loss = loss + self.ps[i](s, a, r, s1, d)
        self.p_optimizer.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.p_params, max_norm=5, norm_type=2)
        self.p_optimizer.step()
        return None
    
    def roll(self, s, a):
        """ batched """
        p = self.ps[np.random.randint(self.n_p)]
        
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = self.act(s, deterministic=False)
                r, s1, d = p(s, a)
            else:
                return None
        return  r, s1, d
    
class MultiAgent(nn.Module):
    def __init__(self, n_agent, **agent_args):
        """
            A wrapper for Multi Agent RL
            Assumes s, a, r, d of shape [b, n_agent, ...]
            (since in general, d may be different for each agent)
        """
        super().__init__()
        agent_fn = agent_args['agent']
        logger = agent_args.pop('logger')
        loggers = logger.fork(n_agent)
        self.agents = []
        for i in range(n_agent):
            self.agents.append(agent_fn(logger = loggers[i], **agent_args))
        self.agents = nn.ModuleList(self.agents)
        
        for attr in ['ps', 'q1', 'pi']:
            if hasattr(self.agents[0], attr):
                setattr(self, attr, True)
        self.pool = None #Pool(n_agent)
        # removed agent parallelism for it does not yield any performance gain
        
    def roll(self, **data):
        data['func'] = 'roll'
        data['agent'] = self.agents
        inputs = dictSplit(data)
        results = parallelEval(self.pool, inputs)
        return listStack(results)
    
    def updateP(self, **data):
        data['func'] = 'updateP'
        data['agent'] = self.agents
        inputs = dictSplit(data)
        results = parallelEval(self.pool, inputs)
        
    def updateQ(self, **data):
        data['func'] = 'updateQ'
        data['agent'] = self.agents
        inputs = dictSplit(data)
        results = parallelEval(self.pool, inputs)

    def updatePi(self, **data):
        data['func'] = 'updatePi'
        data['agent'] = self.agents
        inputs = dictSplit(data)
        results = parallelEval(self.pool, inputs)

    def act(self, S, deterministic=False):
        inputs = dictSplit({'s': S, 'deterministic':deterministic, 'func': 'act', 'agent': self.agents})
        results = parallelEval(self.pool, inputs)
        return torch.stack(results, dim=1)
    
    def setEps(self, eps):
        for agent in self.agents:
            agent.setEps(eps)