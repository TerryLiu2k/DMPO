import numpy as np
import torch
import torch.nn as nn
import itertools
import logging
import time
import os
import pandas as pd
import subprocess

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)


def run_rnn(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)


def one_hot(x, oh_dim, dim=-1):
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma
        self.alpha = alpha
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        nas = np.array(self.adds, dtype=np.int32)
        acts = np.array(self.acts, dtype=np.int32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            # additional spatial rewards
            for t in range(self.max_distance + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.alpha ** t) * rt
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        acts = np.transpose(np.array(self.acts, dtype=np.int32))
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, policies, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = r + self.gamma * cur_R * (1.-done)
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                # additional spatial rewards
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                # additional spatial rewards
                for t in range(max_distance + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val

def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    #cmd = 'sudo cp %s %s' % (src_dir, tar_dir)
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.env.train_mode = True

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            # pre-decision
            policy, action = self._get_policy(ob, done)
            # post-decision
            value = self._get_value(ob, done, action)
            # transition
            self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            self.episode_rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            # collect experience
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            if done:
                break
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent)
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        return ob, done, R

    def perform(self, test_ind, gui=False):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        while not self.global_counter.should_stop():
            # np.random.seed(self.env.seed)
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            while True:
                ob, done, R = self.explore(ob, done)
                dt = self.env.T - self.cur_step
                global_step = self.global_counter.cur_step
                self.model.backward(R, dt, self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    # pytorch implementation is faster, wait SUMO for 1s
                    time.sleep(1)
                    break
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
