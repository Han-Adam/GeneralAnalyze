import copy
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from .network import Actor, Critic


class PPO:
    def __init__(self,
                 path,
                 s_dim=2,
                 a_dim=1,
                 hidden=32,
                 memory_len=50,
                 batch_size=50,
                 update_epoch=10,
                 lr=1e-3,
                 gamma=0.98,
                 lambda_=0.95,
                 epsilon=0.2):
        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden = hidden
        self.memory_len = memory_len
        self.batch_size = batch_size
        self.update_epoch = update_epoch
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.train_it = 0
        self.test = False

        # Network
        self.actor = Actor(s_dim, a_dim, hidden)
        self.actor_old = copy.deepcopy(self.actor)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(s_dim, hidden)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # replay buffer, or memory
        self.memory_s, self.memory_a, self.memory_s_, self.memory_r = [], [], [], []

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            mean, std = self.actor(s)
            if self.test:
                return mean.numpy().tolist()
            else:
                dist = Normal(mean, std)
                a = dist.sample()
                a = torch.clamp(a, -1, 1).numpy().tolist()
                return a

    def store_transition(self, s, a, s_, r):
        self.memory_s.append(s)
        self.memory_a.append(a)
        self.memory_s_.append(s_)
        self.memory_r.append(r)
        if len(self.memory_r) >= self.memory_len:
            s = torch.tensor(self.memory_s, dtype=torch.float)  # [memory_len, s_dim]
            a = torch.tensor(self.memory_a, dtype=torch.float)  # [memory_len, 1(a_dim)]
            r = torch.tensor(self.memory_r, dtype=torch.float)  # [memory_len]
            s_ = torch.tensor(self.memory_s_, dtype=torch.float)
            self._learn(s, a, s_, r)

    def _learn(self, s, a, s_, r):
        self.train_it += 1

        gae = self._gae(s, r, s_)
        r = self._discounted_r(r, s_)

        self.actor_old.load_state_dict(self.actor.state_dict())
        old_log_prob = self._log_prob(s, a, old=True)  # [memory_len, 1]

        for i in range(self.update_epoch):
            for index in range(0, self.memory_len, self.batch_size):
                self._update_actor(s[index: index + self.batch_size],
                                   a[index: index + self.batch_size],
                                   gae[index: index + self.batch_size],
                                   old_log_prob[index: index + self.batch_size])
                self._update_critic(s[index: index + self.batch_size],
                                    r[index: index + self.batch_size])
        # empty the memory
        self.memory_s, self.memory_a, self.memory_s_, self.memory_r, self.memory_done = [], [], [], [], []

    def _gae(self, s, r, s_):
        # calculate the general advantage estimation
        with torch.no_grad():
            v = self.critic(s).squeeze()        # [memory_len]
            v_ = self.critic(s_).squeeze()      # [memory_len]
            delta = r + self.gamma * v_ - v

            length = r.shape[0]
            gae = torch.zeros(size=[length])
            running_add = 0
            for t in range(length - 1, -1, -1):
                gae[t] = running_add * self.gamma * self.lambda_ + delta[t]
                running_add = gae[t]
            return torch.unsqueeze(gae, dim=-1) # [memory_len, 1]

    def _discounted_r(self, r, s_):
        # calculate the discounted reward
        with torch.no_grad():
            length = len(r)
            discounted_r = torch.zeros(size=[length])
            v_ = self.critic(s_)    # [memory_len, 1]
            running_add = v_[length-1]
            for t in range(length - 1, -1, -1):
                discounted_r[t] = running_add * self.gamma + r[t]
                running_add = discounted_r[t]
        return discounted_r.unsqueeze(dim=-1)  # [memory_len, 1]

    def _log_prob(self, s, a, old=False):
        # calculate the log probability
        if old:
            with torch.no_grad():
                mean, std = self.actor_old(s)
        else:
            mean, std = self.actor(s)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(a) # [memory_len, 1]
        return log_prob

    def _update_actor(self, s, a, gae, old_log_prob):
        log_prob = self._log_prob(s, a)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio*gae
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * gae
        loss = -torch.mean(torch.min(surr1, surr2))
        self.opt_actor.zero_grad()
        loss.backward()
        self.opt_actor.step()

    def _update_critic(self, s, r):
        v = self.critic(s)
        loss = F.mse_loss(v, r)
        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

    def store_net(self, prefix):
        torch.save(self.actor.state_dict(), self.path + '/' + prefix + '_Actor.pth')

    def load_net(self, prefix):
        self.actor.load_state_dict(torch.load(self.path + '/' + prefix + '_Actor.pth'))
