import ipdb as pdb
import numpy as np
import torch
import gym
import time
import ray
import random
from tqdm import tqdm, trange
import pickle
from .utils import combined_shape

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Utilizes lazy frames of FrameStack to save memory.
    """

    def __init__(self, max_size, action_dtype):
        self.max_size = max_size
        self.data = []
        self.ptr = 0
        self.unread = 0
        self.action_dtype = action_dtype

    def store(self, obs, act, rew, next_obs, done):
        """ not batched """
        if len(self.data) == self.ptr:
            self.data.append({})
        self.data[self.ptr] = {'s':obs, 'a':act, 'r':rew, 's1':next_obs, 'd':done}
        # lazy frames here
        # cuts Q bootstrap if done (next_obs is arbitrary)
        self.ptr = (self.ptr+1) % self.max_size
        
    def storeBatch(self, obs, act, rew, next_obs, done):
        """ 
            explicitly tell if batched since the first dim may be batch or n_agent
            does not convert to tensor, in order to utilze gym FrameStack LazyFrame
        """
        for i in range(done.shape[0]):
            self.store(obs[i], act[i], rew[i], next_obs[i], done[i])

    def sampleBatch(self, batch_size):
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            if key == 'a':
                dtype = self.action_dtype
            else: # done should be float for convenience
                dtype = torch.float
            lst = [torch.as_tensor(dic[key], dtype=dtype) for dic in raw_batch]
            batch[key] = torch.stack(lst)

        return batch
    
    def iterBatch(self, batch_size):
        """ 
        reads backwards from ptr to use the most recent samples,
        not used because it makes learning unstable
        """
        if self.unread == 0:
            return None
        batch_size =  min(batch_size, self.unread)
        read_ptr = self.ptr - (len(self.data) - self.unread)
        idxs = list(range(read_ptr-batch_size, read_ptr))
        idxs = [(i + batch_size*len(self.data))%len(self.data) for i in idxs] 
        # make them in the correct range
        self.unread -= batch_size
        
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            if key == 'a':
                dtype = self.action_dtype
            else: # done should be float for convenience
                dtype = torch.float
            lst = [torch.as_tensor(dic[key], dtype=dtype) for dic in raw_batch]
            batch[key] = torch.stack(lst)
        return batch
    
    def clear(self):
        self.data = []
        self.ptr = 0
        self._rewind()
        
    def _rewind(self):
        self.unread = len(self.data)


class RL(object):
    def __init__(self, logger, run_args, env_fn, agent_args,
        n_warmup, batch_size, replay_size, 
       max_ep_len, test_interval, n_step, n_test,
       **kwargs):
        """ 
        a generic algorithm for single agent model-based actor-critic, 
        can also be used for model-free, actor-free or crtici-free RL
        For MARL, it is better to overload the agent into a meta-agent instead of overloading RL
        warmup:
            model, q, and policy each warmup for n_warmup steps before used
        """
        self.env, self.test_env = env_fn(), env_fn()
        
        agent = agent_args.agent(logger=logger, run_args=run_args, env=self.env, **agent_args._toDict())
        if not run_args.init_checkpoint is None:
            agent.load(run_args.init_checkpoint)
            logger.log(interaction=run_args.start_step)   
        self.name = run_args.name
        self.start_step = run_args.start_step

        s, self.episode_len, self.episode_reward = self.env.reset(), 0, 0
        self.agent_args = agent_args
        self.p_args, self.pi_args, self.q_args = agent_args.p_args, agent_args.pi_args, agent_args.q_args
        self.agent = agent
        
        self.batch_size = batch_size
        self.n_step = n_step
        self.max_ep_len = max_ep_len

        self.logger = logger
        
        self.test_interval = test_interval
        self.n_test = n_test
        
        # Experience buffer
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dtype = torch.long
        else:
            action_dtype = torch.float
            
        self.env_buffer = ReplayBuffer(max_size=replay_size, action_dtype=action_dtype)
        if not self.p_args is None: # use the model buffer if there is a model
            self.buffer = ReplayBuffer(max_size=self.p_args.model_buffer_size, action_dtype=action_dtype)
        else:
            self.buffer = self.env_buffer  

        # warmups
        self.n_warmup = n_warmup
        # warmup the model before updating pi and q

        # update frequency
        p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args
        # multiple gradient steps per sample if model based RL
        self.q_update_steps = 1
        self.pi_update_steps = 1
        if not self.p_args is None:
            self.p_update_steps = 1
            self.p_update_steps_warmup = 1
            self.branch = agent_args.p_args.branch
            self.refresh_interval = self.agent_args.p_args.refresh_interval
            self.p_update_interval = p_args.update_interval
            self.p_update_interval_warmup = p_args.update_interval_warmup
            if self.p_update_interval < 1:
                self.p_update_steps = int(1/self.p_update_interval)
                self.p_update_interval = 1
            if self.p_update_interval_warmup < 1:
                self.p_update_steps_warmup = int(1/self.p_update_interval_warmup)
                self.p_update_interval_warmup = 1

        if not self.pi_args is None:
            self.pi_update_interval = pi_args.update_interval
            if self.pi_update_interval < 1:
                self.pi_update_steps = int(1/self.pi_update_interval)
                self.pi_update_interval = 1

        self.q_update_interval = q_args.update_interval
        if self.q_update_interval < 1:
            self.q_update_steps = int(1/self.q_update_interval)
            self.q_update_interval = 1

    def test(self):
        returns = []
        scaled = []
        lengths = []
        episodes = []
        for i in trange(self.n_test):
            episode = []
            test_env = self.test_env
            test_env.reset()
            d, ep_ret, ep_len = np.array([False]), 0, 0
            while not(d.any() or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                state = torch.as_tensor(test_env.state, dtype=torch.float)
                action = self.agent.act(state.unsqueeze(0), deterministic=True).squeeze(0)
                # [b=1, (n_agent), ...]
                _, r, d, _ = test_env.step(action.cpu().numpy().squeeze())
                episode += [(test_env.state.tolist(), action.numpy().tolist(), r.tolist())]
                d=np.array(d)
                ep_ret += r.mean()
                ep_len += 1
            if hasattr(test_env, 'rescaleReward'):
                scaled += [ep_ret]
                ep_ret = test_env.rescaleReward(ep_ret, ep_len)
            returns += [ep_ret]
            lengths += [ep_len]
            episodes += [episode]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns, test_episode_len=lengths, test_round=None)
        print(returns)
        print(f"{self.n_test} episodes average accumulated reward: {returns.mean()}")
        if hasattr(test_env, 'rescaleReward'):
            print(f"scaled reward {np.mean(scaled)}")
        with open(f"checkpoints/{self.name}/test.pickle", "wb") as f:
            pickle.dump(episodes, f)
        with open(f"checkpoints/{self.name}/test.txt", "w") as f:
            for episode in episodes:
                for step in episode:
                    f.write(f"{step[0]}, {step[1]}, {step[2]}\n")
                f.write("\n")
        return returns.mean()
        
    def updateAgent(self):
        agent = self.agent
        batch_size = self.batch_size
        env_buffer, buffer = self.env_buffer, self.buffer
        t = self.t
        # Update handling
            
        if not self.p_args is None:
            if t > self.n_warmup:
                p_update_interval = self.p_update_interval
                p_update_steps = self.p_update_steps
            else:
                p_update_interval = self.p_update_interval_warmup
                p_update_steps = self.p_update_steps_warmup
                
            if (t % p_update_interval) == 0 and t>batch_size:
                for i in range(p_update_steps):
                    batch = env_buffer.sampleBatch(batch_size)
                    agent.updateP(**batch)

        if not self.q_args is None and t>self.n_warmup and t % self.q_update_interval == 0:
            for i in range(self.q_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updateQ(**batch)

        if not self.pi_args is None and t>self.n_warmup and t % self.pi_update_interval == 0:
            for i in range(self.pi_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updatePi(**batch)
                
    def roll(self):
        """
            updates the buffer using model rollouts, using the most recent samples in env_buffer
            stops when the buffer is full (max_size + bacthsize -1) or the env_buffer is exhausted
        """
        env_buffer = self.env_buffer
        buffer = self.buffer
        batch_size = self.batch_size
        env_buffer._rewind()
        buffer.clear()
        batch = env_buffer.sampleBatch(self.batch_size)
        while not batch is None and len(buffer.data) < buffer.max_size:
            s = batch['s']
            a = self.agent.act(s)
            for i in range(self.branch):
                r, s1, d = self.agent.roll(s=s)
                buffer.storeBatch(s, a, r, s1, d)
                if len(buffer.data) >= buffer.max_size:
                    break
            batch = env_buffer.sampleBatch(batch_size)
            
    def step(self):
        env = self.env
        state = env.state
        state = torch.as_tensor(state, dtype=torch.float)
        a = self.agent.act(torch.as_tensor(state, dtype=torch.float).unsqueeze(0))    
        a = a.squeeze(0).detach().cpu().numpy()
        # Step the env
        s1, r, d, _ = env.step(a)
        self.episode_reward += r
        self.logger.log(interaction=None)
        self.episode_len += 1
        if self.episode_len == self.max_ep_len:
            """
                some envs return done when episode len is maximum,
                this breaks the Markov property
            """
            d = np.zeros(d.shape, dtype=np.float32)
        d = np.array(d)
        self.env_buffer.store(state, a, r, s1, d)
        if d.any() or (self.episode_len == self.max_ep_len):
            """ for compatibility, allow different agents to have different done"""
            self.logger.log(episode_reward=self.episode_reward.mean(), episode_len=self.episode_len, episode=None)
            _, self.episode_reward, self.episode_len = self.env.reset(), 0, 0
        
    def run(self):
        # Main loop: collect experience in env and update/log each epoch
        last_save = 0
        if self.start_step < self.n_warmup and self.pi_args is None: # eps greedy for q learning
            self.agent.setEps(1)
        else:
            self.agent.setEps(0)
        pbar = iter(tqdm(range(int(self.n_step)), desc=self.name))
        for t in range(self.start_step, self.n_step): 
            next(pbar)
            self.t = t
            
            if t % self.test_interval == 0:
                mean_return = self.test()
                self.agent.save(info = mean_return) # automatically save once per save_period seconds
                
            self.step()
            
            if not self.p_args is None and t >=self.n_warmup \
                and (t% self.refresh_interval == 0 or len(self.buffer.data) == 0):
                self.roll()
                
            if t == self.n_warmup:
                self.agent.setEps(0)
                
            self.updateAgent()        