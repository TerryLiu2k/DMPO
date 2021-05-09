import os
import gym
import random
import numpy as np
import torch
import wandb
import pdb
import time
import dill # overrides multiprocessing

def _runDillEncoded(payload):
    fun, args = dill.loads(payload)
    return fun(args)

def _worker(args):
    """ invokes the agents in parallel"""
    agent = args.pop('agent')
    func = getattr(agent, args.pop('func'))
    result = func(**args)
    return result

def _parallelEval(pool, args):
    payload = [dill.dumps((_worker, item)) for item in args]
    return list(pool.map(_runDillEncoded, payload, chunksize=1))

def parallelEval(pool, args):
    results = []
    for i, arg in enumerate(args):
        agent = arg.pop('agent')
        func = getattr(agent, arg.pop('func'))
        result = func(**arg)
        results.append(result)
    return results

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def dictSelect(dic, idx, dim=1):
    result = {}
    assert dim == 0 or dim ==1
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            if dim == 0:
                result[key] = dic[key][idx]
            else:
                result[key] = dic[key][:,idx]
        elif isinstance(dic[key], torch.nn.ModuleList):
            result[key] = dic[key][idx]
        else:
            result[key] = dic[key]
            
    return result

def dictSplit(dic, dim=1):
    """
        scatters every tensor and modulelist
        others are broadcasted
    """
    results = []
    assert dim == 0 or dim ==1
    sample = dic[list(iter(dic.keys()))[0]]
    length = sample.shape[dim]
    for i in range(length):
        tmp = dictSelect(dic, i, dim)
        results.append(tmp)
    return results

def listStack(lst, dim=1):
    """ 
    takes a list (agent parallel) of lists (return values) and stacks the outer lists
    """
    results = []
    for i in range(len(lst[0])):# torch.stack squeezes..
        results.append(torch.stack([agent_return[i] for agent_return in lst], dim=dim))
    return results

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

class Config(object):
    def __init__(self):
        return None
    def _toDict(self, recursive=False):
        """
            converts to dict for **kwargs
            recursive for logging
        """
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('_') and not name.endswith('_'):
                if isinstance(value, Config) and recursive:
                    value = value._toDict(recursive)
                pr[name] = value
        return pr
    
class TabularLogger(object):
    """
    A text interface logger, outputs mean and std several times per epoch
    """
    def __init__(self):
        self.buffer = {}
        
    def log(dic, commit=False):
        if commit:
            print
        
class Logger(object):
    """
    A logger wrapper with buffer for visualized logger backends, such as tb or wandb
    counting
        all None valued keys are counters
        this feature is helpful when logging from model interior
        since the model should be step-agnostic
    economic logging
        stores the values, log once when flush is called
    syntactic sugar
        supports both .log(data={key: value}) and .log(key=value) 
    custom x axis (wandb is buggy about this)
    multiagent multiprocess logger
        rank = 0 is the algo logger
        rank =1 ,... are the agent loggers
        children get n_interaction from parent
    """
    def __init__(self, args, mute=False, rank=0, parent=None):
        if not mute:
            group = f"{args.algo_args.env_fn.__name__}_{args.algo_args.agent_args.agent.__name__}"
            name = f"{group}_{rank}" if rank else group
            run=wandb.init(
                project="RL",
                config=args._toDict(recursive=True),
                name=name,
                group=group,
            )
            self.logger = run
        self.mute = mute
        self.args = args
        self.step_key = 'interaction'
        self.buffer = {self.step_key: 0}
        self.parent = parent
        self.rank = rank
        self.log_period = args.log_period
        self.save_period = args.save_period
        self.last_save = time.time()
        self.last_log = time.time()

    def fork(self, n):
        loggers = [Logger(self.args, mute=self.mute, rank=i, parent=self) for i in range(n)]
        return loggers
        
    def save(self, model):
        if self.rank is 0 and time.time() - self.last_save >= self.save_period:
            exists_or_mkdir(f"checkpoints/{self.args.name}")
            filename = f"{self.buffer[self.step_key]}.pt"
            if not self.mute:
                with open(f"checkpoints/{self.args.name}/{filename}", 'wb') as f:
                    torch.save(model.state_dict(), f)
                print(f"checkpoint saved as {filename}")
            else:
                print("not saving checkpoints because the logger is muted")
            self.last_save = time.time()
        
    def log(self, raw_data=None, rolling=None, **kwargs):
        if raw_data is None:
            raw_data = {}
        raw_data.update(kwargs)
        
        data = {}
        for key in raw_data: # also logs the mean for histograms
            data[key] = raw_data[key]
            if isinstance(data[key], torch.Tensor) and len(data[key].shape)>0:
                data[key+'_mean'] = data[key].mean()
            
        if self.rank > 0:
            data[self.step_key] = self.parent.buffer[self.step_key]
            
        for key in data:
            if data[key] is None:
                if not key in self.buffer:
                    self.buffer[key] = 0
                self.buffer[key] += 1
            else:
                valid = True
                # check nans
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].detach().cpu()
                    if torch.isnan(data[key]).any():
                        valid = False
                elif np.isnan(data[key]).any():
                    valid = False
                if not valid:
                    print(f'{key} is nan!')
                   # pdb.set_trace()
                    continue
                if rolling and key in self.buffer:
                    self.buffer[key] = self.buffer[key]*(1-1/rolling) + data[key]/rolling
                else:
                    self.buffer[key] = data[key]
        
        if not self.mute and self.rank is 0 and time.time()>self.log_period+self.last_log:
            self.logger.log(data=self.buffer, step =self.buffer[self.step_key], commit=True)
            self.last_log = time.time()

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True