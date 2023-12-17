import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(torch.tanh(x))


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.mean = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                  nn.Tanh(),
                                  nn.Linear(hidden, hidden, bias=False),
                                  nn.Tanh(),
                                  nn.Linear(hidden, a_dim, bias=False))

        self.log_std = nn.Parameter(torch.ones([a_dim]), requires_grad=True)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s):
        mean = self.mean(s)
        std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max).exp()
        std = torch.ones_like(mean) * std
        return mean, std


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False))

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        q2 = self.q2(s_a)
        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        return q1
