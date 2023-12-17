import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self):
        self.pos = 0
        self.vel = 0
        self.target = 0
        self.acc_bound = 5
        self.time_step = 0.01

    def reset(self, pos=None, vel=0, target=0, acc_bound=5):
        self.pos = np.random.rand()*10 - 5 if pos is None else pos
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


class Net():
    def __init__(self):
        n = 3
        print(np.cos(np.pi/n), np.sin(np.pi/n))
        self.w1 = np.array([[np.cos(np.pi/n), np.sin(np.pi/n)],  # [1 / 2 ** 0.5, 1 / 2 ** 0.5],
                            [0, 1]])
        self.w2 = np.array([[2, -1],
                            [2, 0.5]])
        self.w3 = np.array([-1, -1])

    def forward(self, state):
        y1 = np.tanh(np.matmul(self.w1, state))
        y2 = np.tanh(np.matmul(self.w2, y1))
        y3 = np.tanh(np.matmul(self.w3, y2))
        return y3


def main():
    env = Env()
    net = Net()

    N = 301
    bound = 100
    a_record = []
    [x, y] = [0, 0]
    for i in np.linspace(x - bound, x + bound, N):
        state = np.vstack([np.linspace(y - bound, y + bound, N), np.ones(shape=[N]) * i])
        y3 = net.forward(state)
        a_record.append(y3)

    Y, X = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)]
    Z = np.array(a_record)
    X += x
    Y += y
    U = Y
    V = np.array(a_record) * 5
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis')
    fig.colorbar(pcm, ax=ax)
    plt.show()

    length = 3000
    pos = []
    vel = []
    acc = []
    s = env.reset(-80, 0, 0)
    print('start')
    for ep_step in range(length):
        a = net.forward(s)
        s_, r, = env.step(a)
        s = s_
        pos.append(env.pos)
        vel.append(env.vel)
        acc.append(a)
    index = np.array(range(length))
    x = np.linspace(-80, 80, 1000)

    plt.plot(index*0.01, pos)
    plt.plot(index*0.01, vel)
    plt.plot(index*0.01, acc)
    plt.show()


if __name__ == '__main__':
    main()


