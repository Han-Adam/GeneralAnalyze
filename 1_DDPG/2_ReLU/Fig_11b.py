import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Agent import DDPG


def main(gamma, index, net_num, name):
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/DDPG_'+str(gamma)+'/'+str(index)
    agent = DDPG(path, s_dim=2)
    agent.load_net(prefix1='', prefix2=str(net_num))
    agent.var = 0

    N = 1001
    bound = 2000
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
    X /= 1000
    Y /= 1000

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-2, -1, 0, 1, 2])
    y_tick = np.array([-2, -1, 0, 1, 2])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    plt.plot([-0.618, 0, -2], [2, 0, 0.406], c='#AAAAAA', linewidth=0.5)
    fig.savefig('C:/Users/asus/Desktop/DDPG_'+name+'.jpeg', dpi=2000)


if __name__ == '__main__':
    main(0.9, 1, 180, '10')