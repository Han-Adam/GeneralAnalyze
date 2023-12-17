# SAC当中有先sample再tanh的方法，是这个算法独有的
# 其他算法用不了这种先sample再tanh的方法
# SAC也用不了sample过后加clamp的操作。
import copy
import torch
import torch.nn.functional as F
from .network import Actor, Critic
from .replayBuffer import ReplayBuffer
from torch.distributions import Normal


class SAC:
    def __init__(
            self,
            path,
            s_dim=2,
            a_dim=2,
            hidden=32,
            capacity=int(5e4),
            batch_size=512,
            start_learn=512,
            lr=1e-3,
            gamma=0.9,
            tau=5e-3,
            log_prob_reg=1e-6
    ):
        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.start_learn = start_learn
        self.gamma = gamma
        self.tau = tau
        self.log_prob_reg = log_prob_reg

        # Network
        self.actor = Actor(s_dim, a_dim, hidden)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(s_dim, a_dim, hidden)
        self.critic_target = copy.deepcopy(self.critic)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # alpha
        self.target_entropy = -a_dim
        self.alpha = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.alpha], lr=lr)
        # replay buffer, memory
        self.memory = ReplayBuffer(s_dim, a_dim, capacity, batch_size)
        # test
        self.test = False

    def get_action(self, s):
        s = torch.tensor(data=s, dtype=torch.float)
        mean, std = self.actor(s)
        if self.test:
            a = torch.tanh(mean)
        else:
            normal = Normal(mean, std)
            z = normal.rsample()
            a = torch.tanh(z)
        return a.detach().numpy()

    def _log_prob(self, s):
        mean, std = self.actor(s)
        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + self.log_prob_reg)
        log_prob = log_prob.sum(-1, keepdim=True)
        return a, log_prob

    def store_transition(self, s, a, s_, r):
        self.memory.store_transition(s, a, s_, r)
        if self.memory.counter >= self.start_learn:
            s, a, s_, r = self.memory.get_sample()
            self._learn(s, a, s_, r)

    def _learn(self, s, a, s_, r):
        # update q net
        with torch.no_grad():
            a_, log_prob_ = self._log_prob(s_)
            q1_, q2_ = self.critic_target(s_, a_)
            q_target = r + self.gamma * (torch.min(q1_, q2_) - self.alpha*log_prob_)
        q1, q2 = self.critic(s, a)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        q_loss.backward()
        self.opt_critic.step()
        # update policy net
        a_new, log_prob_new = self._log_prob(s)
        q_new = self.critic.Q1(s, a_new)
        # q1_new, q2_new = self.critic(s, a_new)
        # q_new = torch.min(q1_new, q2_new) 这两种做法都可行
        policy_loss = torch.mean(self.alpha*log_prob_new - q_new)
        self.opt_actor.zero_grad()
        policy_loss.backward()
        self.opt_actor.step()
        # update temperature alpha
        alpha_loss = -torch.mean(self.alpha * (log_prob_new+self.target_entropy).detach())
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        # update target net
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def store_net(self, prefix):
        torch.save(self.actor.state_dict(), self.path + '/' + prefix + '_Actor.pth')

    def load_net(self, prefix):
        self.actor.load_state_dict(torch.load(self.path + '/' + prefix + '_Actor.pth'))