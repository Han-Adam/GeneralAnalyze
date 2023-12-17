import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from Agent import SAC
from Env import Env


def main(gamma, index, net_num, name):
    env = Env()
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/SAC_'+str(gamma)+'/'+str(index)
    agent = SAC(path, s_dim=2)
    agent.load_net(prefix=str(net_num))
    agent.test = True

    N = 1001
    bound = 20
    s_record = []
    [x, y] = [0, 0]
    for i in np.linspace(x - bound, x + bound, N):
        state = np.vstack([np.ones(shape=[N]) * i,
                           np.linspace(y - bound, y + bound, N)]).T
        s_record.append(state)
    s_record = np.array(s_record)
    s_record = np.reshape(s_record, newshape=[N**2, 2])
    with torch.no_grad():
        a_record = agent.get_action(s_record)[:, 0]
    a_record = np.reshape(a_record, [N, N])

    X, Y = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)]
    Z = a_record
    X += x
    Y += y

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-20, 20), ylim=(-20, 20))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-20, -10, 0, 10, 20])
    y_tick = np.array([-20, -10, 0, 10, 20])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    fig.savefig('C:/Users/asus/Desktop/SAC_'+name+'.jpeg', dpi=2000)


if __name__ == '__main__':
    main(0.99, 8, 187, '5')
    main(0.99, 6, 83, '10')
    main(0.99, 3, 25, '20')
    main(0.99, 3, 20, '40')


