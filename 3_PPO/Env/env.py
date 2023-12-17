import numpy as np

class Env:
    def __init__(self):
        self.pos = 0
        self.vel = 0
        self.target = 0
        self.acc_bound = 5
        self.time_step = 0.01

    def reset(self, pos=None, vel=0, target=0, acc_bound=5):
        self.pos = np.random.rand()*5 if pos is None else pos
        self.vel = vel
        self.target = target
        self.acc_bound = acc_bound
        return [self.pos-self.target, self.vel]

    def step(self, a):
        r = -np.abs(self.pos - self.target)
        self.vel += a*self.acc_bound * self.time_step
        self.pos += self.vel * self.time_step
        s = [self.pos - self.target, self.vel]
        return s, r