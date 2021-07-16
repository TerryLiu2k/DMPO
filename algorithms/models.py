import numpy as np
import ipdb as pdb
import itertools
from gym.spaces import Box, Discrete
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from .base_util import batch_to_seq, init_layer, one_hot


def MLP(sizes, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def CNN(sizes, kernels, strides, paddings, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Conv2d(sizes[j], sizes[j + 1], kernels[j], strides[j], paddings[j]), act()]
    return nn.Sequential(*layers)


class ParameterizedModel(nn.Module):
    """
        assumes parameterized state representation
        we may use a gaussian prediciton,
        but it degenrates without a kl hyperparam
        unlike the critic and the actor class, 
        the sizes argument does not include the dim of the state
        n_embedding is the number of embedding modules needed, = the number of discrete action spaces used as input
    """

    def __init__(self, env, logger, n_embedding=1, to_predict="srd", gaussian=False, **net_args):
        super().__init__()
        self.logger = logger.child("p")
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        input_dim = net_args['sizes'][0]
        output_dim = net_args['sizes'][-1]
        self.n_embedding = n_embedding
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n, input_dim // n_embedding)
        self.net = MLP(**net_args)
        self.state_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.reward_head = nn.Linear(output_dim, 1)
        self.done_head = nn.Linear(output_dim, 1)
        self.variance_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.NLL = nn.GaussianNLLLoss(reduction='none')
        self.to_predict = to_predict
        self.gaussian = gaussian

    def forward(self, s, a, r=None, s1=None, d=None):
        embedding = s
        if isinstance(self.action_space, Discrete):
            batch_size, _ = a.shape
            action_embedding = self.action_embedding(a).view(batch_size, -1)
            embedding = embedding + action_embedding
        embedding = self.net(embedding)
        state = self.state_head(embedding)
        state_size = state.size()
        reward = self.reward_head(embedding).squeeze(1)
        if self.gaussian:
            variance = self.variance_head(embedding)
            sq_variance = variance.square()

        if r is None:  # inference
            with torch.no_grad():
                done = torch.sigmoid(self.done_head(embedding))
                done = torch.cat([1 - done, done], dim=1)
                done = Categorical(done).sample()  # [b]
                if self.gaussian:
                    state = torch.normal(state, variance.square())
                return reward, state, done

        else:  # training
            done = self.done_head(embedding).squeeze(1)
            if not self.gaussian:
                state_loss = self.MSE(state, s1)
                state_loss = state_loss.mean(dim=1)
                state_var = self.MSE(s1, s1.mean(dim=0, keepdim=True).expand(*s1.shape))
                # we assume the components of state are of similar magnitude
                rel_state_loss = state_loss.mean() / state_var.mean()
                self.logger.log(rel_state_loss=rel_state_loss)
            else:
                state_loss = self.NLL(state, s1, sq_variance.square())
                if state_loss.dim() > 1:
                    state_loss = state_loss.mean(dim=1)
                self.logger.log(state_nll_loss=state_loss, var_mean=sq_variance.square().mean())

            loss = state_loss
            if 'r' in self.to_predict:
                reward_loss = self.MSE(reward, r)
                loss = loss + reward_loss
                reward_var = self.MSE(reward, reward.mean(dim=0, keepdim=True).expand(*reward.shape)).mean()

                self.logger.log(reward_loss=reward_loss,
                                reward_var=reward_var,
                                reward=r)

            if 'd' in self.to_predict:
                done_loss = self.BCE(done, d)
                loss = loss + 10 * done_loss
                done = done > 0
                done_true_positive = (done * d).mean()
                d = d.mean()
                self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

            return (loss, state.detach())


class QCritic(nn.Module):
    """
    Dueling Q, currently only implemented for discrete action space
    if n_embedding > 0, assumes the action space needs embedding
    Notice that the output shape should be 1+action_space.n for discrete dueling Q
    
    n_embedding is the number of embedding modules needed, = the number of discrete action spaces used as input
    only used for decentralized multiagent, assumes the first action is local (consistent with gather() in utils)
    """

    def __init__(self, env, n_embedding=0, **q_args):
        super().__init__()
        q_net = q_args['network']
        self.action_space = env.action_space
        self.q = q_net(**q_args)
        self.n_embedding = n_embedding
        input_dim = q_args['sizes'][0]
        self.state_per_agent = input_dim // (n_embedding + 1)
        if n_embedding != 0:
            self.action_embedding = nn.Embedding(self.action_space.n, self.state_per_agent)

    def forward(self, state, output_distribution, action=None):
        """
        action is only used for decentralized multiagent
        """
        if isinstance(self.action_space, Box):
            q = self.q(torch.cat([state, action], dim=-1))
        else:
            if self.n_embedding > 0:
                # multiagent
                batch_size, _ = action.shape
                action_embedding = self.action_embedding(action).view(batch_size, -1)
                action_embedding[:, :self.state_per_agent] = 0
                state = state + action_embedding
                action = action[:, 0]
            q = self.q(state)
            while len(q.shape) > 2:
                q = q.squeeze(-1)  # HW of size 1 if CNN
            # [b, a+1]
            v = q[:, -1:]
            q = q[:, :-1]
            # q = q - q.mean(dim=1, keepdim=True) + v
            if output_distribution:
                # q for all actions
                return q
            else:
                # q for a particular action
                q = torch.gather(input=q, dim=1, index=action.unsqueeze(-1))
                return q.squeeze(dim=1)


class CategoricalActor(nn.Module):
    """ 
    always returns a distribution
    """

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
        self.eps = 1e-5
        # if pi becomes truely deterministic (e.g. SAC alpha = 0)
        # q will become NaN, use eps to increase stability 
        # and make SAC compatible with "Hard"ActorCritic

    def forward(self, obs):
        logit = self.network(obs)
        while len(logit.shape) > 2:
            logit = logit.squeeze(-1)  # HW of size 1 if CNN
        probs = self.softmax(logit)
        probs = (probs + self.eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


class RegressionActor(nn.Module):
    """
    determinsitc actor, used in DDPG and TD3
    """

    def __init__(self, **net_args):
        super().__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)

    def forward(self, obs):
        out = self.network(obs)
        while len(out.shape) > 2:
            out = out.squeeze(-1)  # HW of size 1 if CNN
        return out


class SquashedGaussianActor(nn.Module):
    """
    Squashed Gaussian actor used in SAC.
    """

    def __init__(self, **net_args):
        super(SquashedGaussianActor, self).__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)

    def forward(self, obs):
        out = self.network(obs)


class NCMultiAgentPolicy(nn.Module):
    """
    Assume all agents are identical.
    """
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__()
        self.name = 'nc'
        self.n_s = n_s
        self.n_a = n_a
        self.neighbor_mask = neighbor_mask  # np array
        self.n_agent = neighbor_mask.shape[0]
        self.n_fc = n_fc
        self.n_h = n_h
        self.n_agent = n_agent
        self.n_step = n_step
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical = identical
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        # h dim: NxTxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i = torch.index_select(p, 0, js)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        return torch.cat(s_i, dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    s_i = self._get_comm_s(i, n_n, x, h, p)
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                h_i, c_i = h[i].unsqueeze(0) * (1-done), c[i].unsqueeze(0) * (1-done)
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach().numpy())
            else:
                vs.append(v_i)
        return vs

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


