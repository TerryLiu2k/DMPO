import os
import gym
import numpy as np
import random
import torch
import wandb
import pdb
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

def parallelEval(pool, args):
    payload = [dill.dumps((_worker, item)) for item in args]
    return list(pool.map(_runDillEncoded, payload, chunksize=1))

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

def listStack(lst, dim=0):
    """ 
    takes a list (agent parallel) of lists (return values) and stacks the outer lists
    .stack inserts a new dim *after* dim
    """
    results = []
    for i in range(len(lst[0])):
        results += [torch.stack([item[i] for item in lst], dim=dim)]
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
    """
    def __init__(self, args, mute=False):
        if not mute:
            run=wandb.init(
                project="RL",
                config=args._toDict(recursive=True),
                name=args.name,
                group=args.env_name,
            )
            self.logger = run
        self.mute = mute
        self.args = args
        self.buffer = {}
        self.step_key = 'interaction'
        
    def save(self, model):
        exists_or_mkdir(f"checkpoints/{self.args.name}")
        filename = f"{self.buffer[self.step_key]}.pt"
        with open(f"checkpoints/{self.args.name}/{filename}", 'wb') as f:
            torch.save(model.state_dict(), f)
        print(f"checkpoint saved as {filename}")
        
    def log(self, raw_data=None, rolling=None, **kwargs):
        if raw_data is None:
            raw_data = {}
        raw_data.update(kwargs)
        
        data = {}
        for key in raw_data: # computes the mean for histograms
            data[key] = raw_data[key]
            if isinstance(data[key], torch.Tensor) and len(data[key].shape)>0:
                data[key+'_mean'] = data[key].mean()
            
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
                
    def flush(self):
        if not self.mute:
            self.logger.log(data=self.buffer, step =self.buffer[self.step_key], commit=True)