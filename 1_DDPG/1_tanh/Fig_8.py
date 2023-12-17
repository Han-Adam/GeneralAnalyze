from Agent import DDPG
from Env import Env
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def main(gamma, index, net_num):
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/DDPG_'+str(gamma)+'/'+str(index)

    agent = DDPG(path, s_dim=2)
    env = Env()

    # test trajectory
    agent.var = 0
    agent.load_net(prefix1='', prefix2=str(net_num))

    # # input-output graph ##############################################################################
    N = 1001
    bound = 100
    s_record = []
    [x, y] = [0, 0]
    for i in np.linspace(x-bound, x+bound, N):
        state = np.vstack([np.ones(shape=[N]) * i,
                           np.linspace(y-bound, y+bound, N)]).T
        s_record.append(state)
    s_record = np.array(s_record)
    s_record = np.reshape(s_record, newshape=[N ** 2, 2])

    with torch.no_grad():
        a_record = agent.get_action(s_record)
    a_record = np.reshape(a_record, [N, N])


    X, Y = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)]
    Z = a_record
    X += x
    Y += y

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])  # (宽，高)
    ax = fig.add_subplot(autoscale_on=False, xlim=(-40, 40), ylim=(-40, 40))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-80, -40, 0, 40, 80])
    y_tick = np.array([-80, -40, 0, 40, 80])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    plt.plot([-500 * 0.63403505/0.77330434, 500*0.63403505/0.77330434], [500, -500], c='#AAAAAA', linewidth=0.5)
    fig.savefig('C:/Users/asus/Desktop/DDPG_1.jpeg', dpi=2000)
    # # input-output graph ##############################################################################

    # # input-output zoom graph #################################################################
    N = 1001
    bound = 5
    s_record = []
    [x, y] = [-60, 75]
    for i in np.linspace(x - bound, x + bound, N):
        state = np.vstack([np.ones(shape=[N]) * i,
                           np.linspace(y - bound, y + bound, N)]).T
        s_record.append(state)
    s_record = np.array(s_record)
    s_record = np.reshape(s_record, newshape=[N ** 2, 2])

    with torch.no_grad():
        a_record = agent.get_action(s_record)
    a_record = np.reshape(a_record, [N, N])

    X, Y = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)]
    Z = a_record
    X += x
    Y += y


    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])  # (宽，高)
    ax = fig.add_subplot(autoscale_on=False, xlim=(-65, -55), ylim=(70, 80))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-65, -60, -55])
    y_tick = np.array([70, 75, 80])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    plt.plot([-5000 * 0.63403505 / 0.77330434, 5000 * 0.63403505 / 0.77330434], [5000, -5000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-5000 * 0.63403505 / 0.77330434, 5000 * 0.63403505 / 0.77330434], [5000 + 2, -5000 + 2], c='#AAAAAA', linewidth=0.5)
    # plt.plot([-500 * 0.5804671 / 0.8142837, 500 * 0.5804671 / 0.8142837], [500, -500], c='#EE0000', linewidth=0.5)
    fig.savefig('C:/Users/asus/Desktop/DDPG_2.jpeg', dpi=2000)
    # zoom ######################################################################

    # weight vector ##############################################################################
    net = agent.actor
    for name in net.state_dict():
        print(name)
    # print(w)
    w1 = net.state_dict()['actor.0.weight'].numpy()
    w2 = net.state_dict()['actor.2.weight'].numpy()
    w3 = net.state_dict()['actor.4.weight'].numpy()

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    # 子图间距
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    # 设置坐标轴
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1.5, 0, 1.5])
    y_tick = np.array([-1.5, 0, 1.5])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    # s = [w1[29, 1]*1000, -w1[29, 0]*1000]
    # v1 = np.tanh(np.matmul(w1, s))
    # v2 = np.tanh(np.matmul(w2, v1))
    # v3 = np.tanh(np.matmul(w3, v2))
    plt.scatter(w1[:, 0], w1[:, 1], s=0.2, color='#0000C9', marker='.')
    # # plt.scatter([w1[2, 0], w1[5, 0]],
    # #             [w1[2, 1], w1[5, 1]])
    plt.scatter([w1[15, 0]],
                [w1[15, 1]], s=2, color='#7F0000', marker='.')
    plt.plot([0, w1[15, 0]],
             [0, w1[15, 1]], linewidth=0.5, color='#7F0000')
    # plt.plot([0, w1[17, 0]],
    #          [0, w1[17, 1]], linewidth=0.5, color='#7F0000')
    plt.plot([-500 * 0.3139654 / 0.38292965, 500 * 0.3139654 / 0.38292965], [500, -500], c='#AAAAAA', linewidth=0.5)
    # plt.plot([-500 * 0.92591244 / 1.2988771, 500 * 0.92591244 / 1.2988771], [500, -500], c='#EE0000', linewidth=0.5)
    # plt.show()
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/DDPG_3.jpeg', dpi=2000)
    # input-output graph ##############################################################################

    # \bar\phi(15) 15 ##############################################################################
    print(w1[15])
    s = [-w1[15, 1] * 100, w1[15, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    # print(v1)
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[15] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3)

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-6, 6))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    # 设置坐标轴
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([-5, 0, 5])
    plt.xticks(x_tick)
    plt.yticks(y_tick)

    plt.plot(index, np.array(v3_record)*5, linewidth=0.5, color='#007F00')
    plt.scatter([0.55317], [0], s=2, color='#007F00', marker='.')
    plt.grid(color='#999999', ls='--', lw=0.25)
    # plt.show()
    fig.savefig('C:/Users/asus/Desktop/DDPG_4.jpeg', dpi=2000)
    # \bar\phi(15) 15 ##############################################################################

# 0.99, 0, 168, 关键方向：[-0.548625, 1]

if __name__ == '__main__':
    main(0.99, 3, 89)

    # 15： 9.148 -4.998， 4.150
    # 17： 0.8495 4.150 4.999

