import pdb
import numpy as np
import torch
import gym
import time
import random
from tqdm import tqdm
from utils import combined_shape

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Utilizes lazy frames of FrameStack to save memory.
    """

    def __init__(self, max_size, device):
        self.max_size = max_size
        self.data = []
        self.ptr = 0
        self.read_ptr = 0
        self.device = device

    def _store(self, obs, act, rew, next_obs, done):
        """ not batched """
        if len(self.data) == self.ptr:
            self.data.append({})
        self.data[self.ptr] = {'s':obs, 'a':act, 'r':rew, 's1':next_obs, 'd':float(done)}
        # lazy frames here
        # cuts Q bootstrap if done (next_obs is arbitrary)
        self.ptr = (self.ptr+1) % self.max_size
        
    def store(self, obs, act, rew, next_obs, done):
        """ can be batched"""
        if not isinstance(done, bool): # batched
            for i in range(done.shape[0]):
                self._store(obs[i], act[i], rew[i], next_obs[i], done[i])
        else:
            self._store(obs, act, rew, next_obs, done)

        
    def sampleBatch(self, batch_size):
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            try:
                lst = [torch.as_tensor(dic[key]) for dic in raw_batch]
                batch[key] = torch.stack(lst).to(self.device)
            except:
                pdb.set_trace()
        return batch
    
    def iterBatch(self, batch_size):
        if self.read_ptr >= len(self.data):
            self._rewind()
            return None
        
        end =  min(self.read_ptr+batch_size, len(self.data))
        idxs = list(range(self.read_ptr, end))
        self.read_ptr = end
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            try:
                lst = [torch.as_tensor(dic[key]) for dic in raw_batch]
                batch[key] = torch.stack(lst).to(self.device)
            except:
                pdb.set_trace()
        return batch
    
    def clear(self):
        self.data = []
        self.ptr = 0
        self.read_ptr = 0
        
    def _rewind(self):
        self.read_ptr = 0



def RL(logger, device,
       env_fn, agent_args,
        n_warmup, batch_size, replay_size,
       max_ep_len, test_interval, save_interval,
       seed, n_step, log_interval,
       p_update_interval=None, q_update_interval=None, pi_update_interval=None,
       **kwargs):
    """ 
    a generic algorithm for model-free reinforcement learning
    plugin state preprocessing if necessary, by wrapping the env
    warmup:
        model, q, and policy each warmup for n_warmup steps before used
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args
    agent = agent_args.agent(logger=logger, **agent_args._toDict())
    agent = agent.to(device)
    last_save = 0
    
    pbar = iter(tqdm(range(int(1e6))))
    
    # Experience buffer
    env_buffer = ReplayBuffer(max_size=replay_size, device=device)
    if hasattr(agent, "ps"): # use the model buffer if there is a model
        buffer = ReplayBuffer(max_size=replay_size, device=device)
    else:
        buffer = env_buffer  

    # warmups
    if hasattr(agent, "ps"):
        q_update_start = n_warmup 
        # p and q starts at the same time, since q update also need p
        pi_update_start = n_warmup
        act_start = 2*n_warmup
    else:
        q_update_start = 0
        pi_update_start = 0
        act_start = n_warmup
        
    # multiple gradient steps per sample if model based RL
    p_update_steps = 1
    q_update_steps = 1
    pi_update_steps = 1
    if hasattr(agent, "ps"):
        p_update_interval = p_args.update_interval
        if p_update_interval < 1:
            p_update_steps = int(1/p_update_interval)
            p_update_interval = 1
            
    if hasattr(agent, "pi"):
        pi_update_interval = pi_args.update_interval
        if pi_update_interval < 1:
            pi_update_steps = int(1/pi_update_interval)
            pi_update_interval = 1
            
    q_update_interval = q_args.update_interval
    if q_update_interval < 1:
        q_update_steps = int(1/q_update_interval)
        q_update_interval = 1

    def test_agent():
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            action = agent.act(torch.as_tensor(o,  dtype=torch.float).to(device), deterministic=True)
            o, r, d, _ = test_env.step(action.cpu().numpy())
            ep_ret += r
            ep_len += 1
        logger.log(TestEpRet=ep_ret, TestEpLen=ep_len, testEpisode=None)

    # Prepare for interaction with environment
    start_time = time.time()
    env, test_env = env_fn(), env_fn()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(n_step): 
        logger.log(interaction=None)
        if t >= act_start:
            agent.random = False
            
        a = agent.act(torch.as_tensor(o,  dtype=torch.float).to(device))    
        a = a.detach().cpu().numpy().item()
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len==max_ep_len else d
        env_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            logger.log(EpRet=ep_ret, EpLen=ep_len, episode=None)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # model rollout
        if hasattr(agent, "ps") and t% p_args.refresh_interval == 0 and t >=n_warmup:
            env_buffer._rewind()
            buffer.clear()
            batch = env_buffer.iterBatch(batch_size)
            while not batch is None and len(buffer.data) < buffer.max_size:
                s = batch['s']
                a = agent.act(s, batched=True)
                for i in range(p_args.branch):
                    r, s1, d = agent.roll(s, a)
                    buffer.store(s, a, r, s1, d)
                batch = env_buffer.iterBatch(batch_size)
                        
        # Update handling
        if hasattr(agent, "ps")  and (t % p_update_interval) == 0 and t>batch_size:
            for i in range(p_update_steps):
                batch = env_buffer.sampleBatch(batch_size)
                agent.updateP(data=batch)
            
        if hasattr(agent, "q1") and t>q_update_start and t % q_update_interval == 0:
            for i in range(q_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updateQ(data=batch)
            
        if hasattr(agent, "pi") and t>pi_update_start and t % pi_update_interval == 0:
            for i in range(pi_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updatePi(data=batch)
                
        if time.time() - last_save >= save_interval:
            logger.save(agent)
            last_save = time.time()
            
        if (t) % test_interval == 0:
            test_agent()
                
        # End of epoch handling
        if t % log_interval == 0:
            next(pbar)
            # Test the performance of the deterministic version of the agent.
            # Log info about epoch
            logger.log(epoch=None)
            logger.flush()
    
    return ac