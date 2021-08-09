import time
from copy import deepcopy

import torch
import torch.optim as optim
from torch.multiprocessing import Pool, Process, set_start_method
import logging
import ray
import os
from .base_util import MultiAgentOnPolicyBuffer, Scheduler
from .utils import dictSelect, dictSplit, listStack, parallelEval, sequentialEval, locate
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
        s, a = locate(self.alpha.device, s, a)
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
        self.logger.log(q_update=None, q_loss=loss_q/2, reward = r)
        
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
        if target_entropy is None:
            self.alpha.requires_grad = False
        self.target_entropy = target_entropy
        
        self.eps = 0
        self.action_space = env.action_space
        if isinstance(self.action_space, Box): #continous
            self.pi = SquashedGaussianActor(**pi_args._toDict())
        else:
            self.pi = CategoricalActor(**pi_args._toDict())
                                
        if target_entropy is not None:
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
            #loss = regret.mean()
            loss = -(pi*q).sum(dim=1).mean()
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            if self.target_entropy is not None:
                alpha_loss = (entropy.detach()-self.target_entropy)*self.alpha
                loss = loss + alpha_loss
            self.logger.log(pi_entropy=entropy, pi_regret=regret.mean(), alpha=self.alpha)
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
            self.logger.log(q_update=None, loss_q=loss_q/2, reward = r)

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
            loss_, s1_ =  self.ps[i](s, a, r, s1, d)
            loss = loss + loss_
        self.p_optimizer.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.p_params, max_norm=5, norm_type=2)
        self.p_optimizer.step()
        return (s1_,)
    
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

@ray.remote
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
    def __init__(self, n_agent, parallel, env, wrappers, run_args, **agent_args):
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
        n_cpu, n_gpu, device = run_args.n_cpu, run_args.n_gpu, run_args.device
        for i in range(n_agent):
            if parallel:
                agent = Worker.options(num_gpus = n_gpu, num_cpus=n_cpu).remote(agent_fn=agent_fn,
                                                                         device=device, logger = logger.child(f"{i}"), env=env, **agent_args)
            else:
                agent = agent_fn(device=device, env=env, logger=logger.child(f"{i}"), **agent_args).to(device)
            self.agents.append(agent)
        if parallel: 
            self.eval = parallelEval
        else:
            self.eval = sequentialEval
        wrappers['p_out'] = listStack
        # (s, r, d)
        if not wrappers.__contains__('q_out') or wrappers['q_out'] is None:
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
        results = self.eval(self.agents, 'roll' ,data)
        results = self.wrappers['p_out'](results) # r, s1, d
        if not 'r' in self.p_to_predict:
            s1 = results[1]
            reward, done = self.env.state2Reward(s1)
            return reward, s1, done
        return results
    
    def updateP(self, **data):
        reward = data['r']
        data_split = self.wrappers['p_in'](data)
        results = self.eval(self.agents, 'updateP', data_split)
        # returns r, s1, d for logging
        results = self.wrappers['p_out'](results)
        results = locate('cpu', *results)
        
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
        results = self.eval(self.agents, 'updateQ', inputs)
        
    def _evalQ(self, **data):
        data = {'output_distribution': True, 
                 's': data['s'], 
                 'a': data['a']}
        data = self.wrappers['q_in'](data)
        results = self.eval(self.agents, '_evalQ', data)
        results = self.wrappers['q_out'](results)
        return results
        
    def updatePi(self, **data):
        q = self._evalQ(**data)
        # a list of qs
        data = {'q': q,
                's': data['s']
               }
        data = self.wrappers['pi_in'](data)
        results = self.eval(self.agents, 'updatePi', data)    

    def act(self, s, deterministic=False, output_distribution=False):
        data = {'s': s, 'deterministic':deterministic, 
               'output_distribution': output_distribution}
        inputs = self.wrappers['pi_in'](data)
        results = self.eval(self.agents, 'act', inputs)
        if output_distribution:
            return listStack(results)
        else:
            return torch.stack(results, dim=1)
    
    def setEps(self, eps):
        self.eval(self.agents, 'setEps', [{'eps': eps}]*len(self.agents))
        
    def save(self, info=None):
        self.eval(self.agents, 'save', [{'info': info}]*len(self.agents))
        ray.get(self.logger.server.save.remote(flush=True))
        
    def load(self, path):
        with open(path, "rb") as file:
            dic = torch.load(file)
        self.eval(self.agents, 'load', [{'state_dict': dic}]*len(self.agents))
        print(f"checkpointed loaded from {path}")


class NeurComm:
    """
    Implementation of NeurComm, maximum code re-utilization.
    Epsilon-greedy is not implemented, meaning epsilon has no effect.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, p, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.identical_agent:
            self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                             reward, value, done)
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)
        self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                             summary_writer=summary_writer, global_step=global_step)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, ps, actions=None, out_type='p'):
        if self.identical_agent:
            return self.policy.forward(np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def reset(self):
        self.policy._reset()

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have varient dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        # init torch
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('Use gpu for pytorch...')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('Use cpu for pytorch...')

        self.policy = self._init_policy()
        self.policy.to(self.device)

        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init,
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)

    def _update_lr(self):
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p


class NeurCommWrapper(NeurComm):
    def __init__(self, env, logger, run_args, agent_config, seed=None, **args):
        if seed is None:
            seed = int(time.time()*1000) % 65536
        self.logger = logger
        self.name = 'ma2c_nc'
        self.eps = 0.0
        super().__init__(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                   run_args.n_step, agent_config['MODEL_CONFIG'], seed=seed)

    def setEps(self, eps):
        self.eps = eps

    def act(self, s, deterministic=False):
        """Parameter deterministic has no effect."""
        return self.forward(s, np.array([False]*self.n_agent), )

class SAC_New(SAC):
    def __init__(self, logger, env, q_args, pi_args, gamma, target_entropy, target_sync_rate, alpha=0, **kwargs):
        super(SAC_New, self).__init__(logger, env, q_args, pi_args, gamma, target_entropy, target_sync_rate, alpha=0, **kwargs)
        self.q1 = QCritic_New(env, **q_args._toDict())
        self.q2 = QCritic_New(env, **q_args._toDict())
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

    def updatePi(self, s, q = None):
        s = s.to(self.alpha.device)
        if not q is None:
            q = q.to(self.alpha.device)
        if isinstance(self.action_space, Discrete):
            pi = self.pi(s) + 1e-5  # avoid nan
            logp = torch.log(pi)
            a = Categorical(pi).sample().view(-1, 1)
            pi_a = torch.gather(pi, dim=1, index=a)
            logpi_a = torch.log(pi_a)
            if q is None:
                q1 = self.q1(s)
                q2 = self.q2(s)
                q = torch.min(q1, q2)
            q = q - self.alpha.detach()
            #q = q - self.alpha.detach() * logp
            #optimum = q.max(dim=1, keepdim=True)[0].detach()
            #regret = optimum - (pi * q).sum(dim=1)
            # loss = regret.mean()
            #loss = -(pi * q).sum(dim=1).mean()
            loss = -(q * logpi_a).sum(dim=1).mean(dim=0)
            entropy = -(pi * logp).sum(dim=1).mean(dim=0)
            if self.target_entropy is not None:
                alpha_loss = (entropy.detach() - self.target_entropy) * self.alpha
                loss = loss + alpha_loss
            #self.logger.log(pi_entropy=entropy, pi_regret=regret.mean(), alpha=self.alpha)
            self.logger.log(pi_entropy=entropy, alpha=self.alpha)
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
                entropy = -(p_a1 * loga1).sum(dim=1, keepdim=True)
                q1_pi_targ = self.q1_target(s1, False, a1)
                q2_pi_targ = self.q2_target(s1, False, a1)  # [b, n_a]
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha.detach() * entropy
                #q_pi_targ = (p_a1 * q_pi_targ).sum(dim=1)
                backup = r + self.gamma * (1 - d) * (q_pi_targ).squeeze()

            # MSE loss against Bellman backup
            loss_q1 = ((q1.squeeze() - backup) ** 2).mean()
            loss_q2 = ((q2.squeeze() - backup) ** 2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            self.logger.log(q=q_pi_targ, q_diff=((q1 + q2) / 2 - backup).mean())

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer.zero_grad()
            if not torch.isnan(loss_q).any():
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
                self.q_optimizer.step()

            # Record things
            self.logger.log(q_update=None, loss_q=loss_q / 2, reward=r)

            # update the target nets
            with torch.no_grad():
                for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                    for p, p_targ in zip(current.parameters(), target.parameters()):
                        p_targ.data.mul_(1 - self.target_sync_rate)
                        p_targ.data.add_(self.target_sync_rate * p.data)

class MBPO_New(SAC_New):
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
            loss_, s1_ =  self.ps[i](s, a, r, s1, d)
            loss = loss + loss_
        self.p_optimizer.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.p_params, max_norm=5, norm_type=2)
        self.p_optimizer.step()
        return (s1_,)
    
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

