import os
import gym
import random
import numpy as np
import torch
import wandb
from ray.util import pdb
import ray
import time
from torch.utils.tensorboard import SummaryWriter

def parallelEval(agents, func, args):
    """
    expects a list of dicts
    """
    results = []
    for i, arg in enumerate(args):
        agent = agents[i]
        remote = getattr(agent, func).remote
        result = remote(**arg)
        results.append(result)
    results = ray.get(results)
    return results

def gather(k):
    def _gather(tensor):
        """ 
        for multiple agents aligned along an axis to collect information from their k-hop neighbor
        input: [b, n_agent, dim], returns [b, n_agent, dim*n_reception_field]
        action is an one-hot embedding
        
        the first is local
        """
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape

        result = torch.zeros((b, n, (1+2*k)*depth), dtype = tensor.dtype, device=tensor.device)
        for i in range(n):
            for j in range(i-k, i+k+1):
                if j<0 or j>=n:
                    continue
                start = (j-i +1 +2*k)%(1+2*k)
                result[:, i, start*depth: start*depth+depth] = tensor[:, j]
        return result
    if k > 0:
        return _gather
    else:
        return lambda x: x
    
def reduce(k):
    def _reduce(tensor):
        """ 
        for multiple agents aligned along an axis to collect information from their k-hop neighbor
        input: [b, n_agent, dim], returns [b, n_agent, dim*n_reception_field]
        action is an one-hot embedding
        
        the first is local
        """
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape

        result = torch.zeros((b, n, depth), dtype = tensor.dtype, device=tensor.device)
        for i in range(n):
            for j in range(i-k, i+k+1):
                if j<0 or j>=n:
                    continue
                result[:, i] += tensor[:, j]
        return result
    if k > 0:
        return _reduce
    else:
        return lambda x: x
    
def collect(dic={}):
    """
    selects a different gather radius (more generally, collective operation) for each data key
    the wrapper inputs raw, no redundancy data from the env
    outputs a list containing data for each agent
    """
    def wrapper(data):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                if key in dic:
                    data[key] = dic[key](data[key])
                elif "*" in dic:
                    data[key] = dic["*"](data[key])
        return dictSplit(data)
    return wrapper

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
        gathers every tensor and modulelist
        others are broadcasted
    """
    results = []
    assert dim == 0 or dim ==1
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            length = dic[key].shape[dim]
            break
    for i in range(length):
        tmp = dictSelect(dic, i, dim)
        results.append(tmp)
    return results

def listStack(lst, dim=1):
    """ 
    takes a list (agent parallel) of lists (return values) and stacks the outer lists
    """
    results = []
    for i in range(len(lst[0])):
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
    
class LogClient(object):
    """
    A logger wrapper with buffer for visualized logger backends, such as tb or wandb
    counting
        all None valued keys are counters
        this feature is helpful when logging from model interior
        since the model should be step-agnostic
    economic logging
        stores the values, log once per log_period
    syntactic sugar
        supports both .log(data={key: value}) and .log(key=value) 
    multiple backends
        forwards logging to both tensorboard and wandb
    logger hiearchy and multiagent multiprocess logger
        the prefix does not end with "/"
        prefix = "" is the root logger
        prefix = "*/agent0" ,... are the agent loggers
        children get n_interaction from the root logger
    """
    def __init__(self, server, prefix=""):
        self.buffer = {}
        self.prefix = prefix
        if isinstance(server, LogClient):
            server = server.server
            prefix = f"{server.prefix}/{prefix}"
        self.log_period = server.args.log_period
        self.last_log = 0
        
    def child(self, prefix=""):
        return LogClient(prefix, self)
        
    def flush(self):
        ray.get(self.server.flush.remote(self))
        self.last_log = time.time()
        
    def log(self, raw_data=None, rolling=None, **kwargs):
        if raw_data is None:
            raw_data = {}
        raw_data.update(kwargs)
        
        data = {}
        for key in raw_data: # also logs the mean for histograms
            data[key] = raw_data[key]
            if isinstance(data[key], torch.Tensor) and len(data[key].shape)>0 or\
            isinstance(data[key], np.ndarray) and len(data[key].shape)> 0:
                data[key+'_mean'] = data[key].mean()
            
        # updates the buffer
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

        # uploading
        if not self.mute and time.time()>self.log_period+self.last_log:
            self.flush()

    def save(self, model, info=None):
        ray.get(self.server.save.remote({self.prefix: model.state_dict()}, info))

            

@ray.remote
class LogServer(object):
    """
    We do not assume the logging backend (e.g. tb, wandb) supports multiprocess logging,
    therefore we implement a centralized log manager
    
    It should not be directly invoked, since we want to hide the implementation detail (.log.remote)
    Wrap it with prefix="" to get the root logger
    """
    def __init__(self, args, mute=False):
        self.group = args.algo_args.env_fn.__name__
        self.name = f"{self.group}_{args.algo_args.agent_args.agent.__name__}_{args.seed}"
        args.name = self.name
        if not mute:
            if root is None:
                run=wandb.init(
                    project="RL",
                    config=args._toDict(recursive=True),
                    name=self.name,
                    group=self.group,
                )
                self.logger = run
                self.writer = SummaryWriter(log_dir=f"runs/{self.name}")
                self.writer.add_text("config", f"{args._toDict(recursive=True)}")
            else:
                self.logger = root.logger
                self.writer = root.writer
        self.mute = mute
        self.args = args
        self.save_period = args.save_period
        self.last_save = time.time()
        self.state_dict = {}
            
    def flush(self, logger=None):
        if logger = None:
            logger = self
        buffer = logger.buffer
        data = {}
        for key in buffer:
            log_key = logger.prefix+"/"+key
            while log_key[0] == '/':
                 # removes the first slash, to be wandb compatible
                log_key = log_key[1:]
            data[log_key] = buffer[key]

            if isinstance(data[log_key], torch.Tensor) and len(data[log_key].shape)>0 or\
            isinstance(data[log_key], np.ndarray) and len(data[log_key].shape)> 0:
                self.writer.add_histogram(log_key, data[log_key], self.buffer[self.step_key])
            else:
                self.writer.add_scalar(log_key, data[log_key], self.buffer[self.step_key])

        self.logger.log(data=data, step =self.buffer[self.step_key], commit=False)
        # "warning: step must only increase "commit = True
        # because wandb assumes step must increase per commit
        self.last_log = time.time()
        
    def save(self, state_dict, info=None):
        self.state_dict.update(state_dict)
        if time.time() - self.last_save >= self.save_period:
            exists_or_mkdir(f"checkpoints/{self.name}")
            filename = f"{self.buffer[self.step_key]}_{info}.pt"
            if not self.mute:
                with open(f"checkpoints/{self.name}/{filename}", 'wb') as f:
                    torch.save(self.state_dict, f)
                print(f"checkpoint saved as {filename}")
            else:
                print("not saving checkpoints because the logger is muted")
            self.last_save = time.time()
            

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True