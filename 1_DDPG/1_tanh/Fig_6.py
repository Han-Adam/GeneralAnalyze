import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Agent import DDPG
from Env import Env


def main(gamma, index, net_num, name):
    env = Env()
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/DDPG_'+str(gamma)+'/'+str(index)
    agent = DDPG(path, s_dim=2)
    agent.load_net(prefix1='', prefix2=str(net_num))
    agent.var = 0

    # response 1
    length = 2000
    pos = []
    vel = []
    acc = []
    s = env.reset(0, 0, 5)
    for ep_step in range(length):
        a = agent.get_action(s)
        s_, r, = env.step(a[0])
        s = s_
        pos.append(env.pos - env.target)
        vel.append(env.vel)
        acc.append(a)
    response1 = np.vstack([pos, vel])
    # response 2
    length = 2000
    pos = []
    vel = []
    acc = []
    s = env.reset(0, 0, 10)
    for ep_step in range(length):
        a = agent.get_action(s)
        s_, r, = env.step(a[0])
        s = s_
        pos.append(env.pos - env.target)
        vel.append(env.vel)
        acc.append(a)
    response2 = np.vstack([pos, vel])
    # response 3
    length = 2000
    pos = []
    vel = []
    acc = []
    s = env.reset(0, 0, 20)
    for ep_step in range(length):
        a = agent.get_action(s)
        s_, r, = env.step(a[0])
        s = s_
        pos.append(env.pos - env.target)
        vel.append(env.vel)
        acc.append(a)
    response3 = np.vstack([pos, vel])
    # response 4
    length = 2000
    pos = []
    vel = []
    acc = []
    s = env.reset(0, 0, 40)
    for ep_step in range(length):
        a = agent.get_action(s)
        s_, r, = env.step(a[0])
        s = s_
        pos.append(env.pos - env.target)
        vel.append(env.vel)
        acc.append(a)
    response4 = np.vstack([pos, vel])

    N = 1001
    bound = 40
    s_record = []
    [x, y] = [0, 0]
    for i in np.linspace(x - bound, x + bound, N):
        state = np.vstack([np.ones(shape=[N]) * i,
                           np.linspace(y - bound, y + bound, N)]).T
        s_record.append(state)
    s_record = np.array(s_record)
    s_record = np.reshape(s_record, newshape=[N**2, 2])
    with torch.no_grad():
        a_record = agent.get_action(s_record)
    a_record = np.reshape(a_record, [N, N])

    X, Y = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)]
    Z = a_record
    X += x
    Y += y

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-40, 40), ylim=(-40, 40))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-40, -20, 0, 20, 40])
    y_tick = np.array([-40, -20, 0, 20, 40])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    plt.plot(response1[0, :], response1[1, :], c='#EE0000', linewidth=0.5)
    plt.plot(response2[0, :], response2[1, :], c='#EE0000', linewidth=0.5)
    plt.plot(response3[0, :], response3[1, :], c='#EE0000', linewidth=0.5)
    plt.plot(response4[0, :], response4[1, :], c='#EE0000', linewidth=0.5)
    fig.savefig('C:/Users/asus/Desktop/DDPG_'+name+'.jpeg', dpi=2000)


if __name__ == '__main__':
    main(0.99, 9, 65, '5')
    main(0.99, 3, 89, '10')  # fig6 DDPG (a)
    main(0.99, 9, 91, '20')  # fig6 DDPG (b)
    main(0.99, 9, 99, '40')  # fig6 DDPG (c)


