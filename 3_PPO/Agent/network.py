import torch.nn as nn
import torch

def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(torch.tanh(x))


class Actor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(Actor, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                     nn.Tanh(),
                                     nn.Linear(hidden, hidden, bias=False),
                                     nn.Tanh())
        self.mean = nn.Sequential(nn.Linear(hidden, a_dim, bias=False),
                                  nn.Tanh())
        # self.std = nn.Sequential(nn.Linear(hidden, a_dim),
        #                          nn.Softplus())
        self.log_std = nn.Parameter(torch.ones(size=[a_dim]), requires_grad=True)
        init_linear(self)

    def forward(self, s):
        feature = self.feature(s)
        mean = self.mean(feature)
        std = self.log_std.exp()
        return mean, std


class Critic(nn.Module):
    def __init__(self, s_dim, hidden):
        super(Critic, self).__init__()
        self.v = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                               Abs(),
                               nn.Linear(hidden, hidden, bias=False),
                               Abs(),
                               nn.Linear(hidden, 1, bias=False))
        init_linear(self)

    def forward(self, s):
        return self.v(s)

