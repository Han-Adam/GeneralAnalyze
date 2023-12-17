from Agent import PPO
from Env import Env
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def main(gamma, index, net_num):
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/PPO_'+str(gamma)+'/'+str(index)

    agent = PPO(path, s_dim=2)
    env = Env()

    # test trajectory
    agent.test = True
    agent.load_net(prefix=str(net_num))

    net = agent.actor
    for name in net.state_dict():
        print(name)
    # print(w)
    w1 = net.state_dict()['feature.0.weight'].numpy()
    w2 = net.state_dict()['feature.2.weight'].numpy()
    w3 = net.state_dict()['mean.0.weight'].numpy()

    # # input-output graph 1 ##############################################################################
    N = 1001
    bound = 2000
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

    X, Y = np.mgrid[-bound:bound:complex(0, N), -bound:bound:complex(0, N)] / 1000
    Z = a_record
    X += x
    Y += y

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
    # 18: 3.688, 15: 2.94, 26: 1.19, 28: 0.8, 14: 0.66, 11: 0.24
    plt.show()
    # fig.savefig('C:/Users/asus/Desktop/PPO_1.jpeg', dpi=2000)
    # # input-output graph 1##############################################################################

    # # input-output graph 2##############################################################################
    N = 1001
    bound = 500
    s_record = []
    [x, y] = [-1700, 1100]
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
    X /= 1000
    Y /= 1000

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, -1.6), ylim=(1, 1.4))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-2, -1.9, -1.8, -1.7, -1.6])
    y_tick = np.array([1, 1.1, 1.2, 1.3, 1.4])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.tick_params(direction='out')
    print(np.array(Z).shape)
    pcm = ax.pcolor(X, Y, Z, cmap='viridis', vmin=-1, vmax=1)
    # 18: 3.688, 15: 2.94, 26: 1.19, 28: 0.8, 14: 0.66, 11: 0.24
    plt.plot([-1000*w1[18, 1]/w1[18, 0], 1000*w1[18, 1]/w1[18, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[15, 1] / w1[15, 0], 1000 * w1[15, 1]/w1[15, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[26, 1] / w1[26, 0], 1000 * w1[26, 1] / w1[26, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[28, 1] / w1[28, 0], 1000 * w1[28, 1] / w1[28, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[14, 1] / w1[14, 0], 1000 * w1[14, 1] / w1[14, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    fig.savefig('C:/Users/asus/Desktop/PPO_2.jpeg', dpi=2000)
    # # input-output graph 2##############################################################################
    #
    # # weight vector ##############################################################################
    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-0.25, 0.25), ylim=(-0.25, 0.25))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-0.2, 0, 0.2])
    y_tick = np.array([-0.2, 0, 0.2])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.scatter(w1[:, 0], w1[:, 1], s=0.2, color='#0000C9', marker='.')
    # 18: 3.688, 15: 2.94, 26: 1.19, 28: 0.8, 14: 0.66, 11: 0.24
    plt.scatter([w1[18, 0], w1[15, 0], w1[26, 0], w1[28, 0], w1[14, 0]],
                [w1[18, 1], w1[15, 1], w1[26, 1], w1[28, 1], w1[14, 1]], s=2, color='#7F0000', marker='.')
    plt.plot([0, w1[18, 0]],
             [0, w1[18, 1]], linewidth=0.5, color='#7F0000')

    plt.plot([0, w1[15, 0]],
             [0, w1[15, 1]], linewidth=0.5, color='#7F0000')

    plt.plot([0, w1[26, 0]],
             [0, w1[26, 1]], linewidth=0.5, color='#7F0000')

    plt.plot([0, w1[28, 0]],
             [0, w1[28, 1]], linewidth=0.5, color='#7F0000')

    plt.plot([0, w1[14, 0]],
             [0, w1[14, 1]], linewidth=0.5, color='#7F0000')

    plt.plot([-1000 * w1[18, 1] / w1[18, 0], 1000 * w1[18, 1] / w1[18, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[15, 1] / w1[15, 0], 1000 * w1[15, 1] / w1[15, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[26, 1] / w1[26, 0], 1000 * w1[26, 1] / w1[26, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[28, 1] / w1[28, 0], 1000 * w1[28, 1] / w1[28, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    plt.plot([-1000 * w1[14, 1] / w1[14, 0], 1000 * w1[14, 1] / w1[14, 0]], [1000, -1000], c='#AAAAAA', linewidth=0.5)
    # plt.plot([0, w1[9, 0]],
    #          [0, w1[9, 1]], linewidth=0.5, color='#7F0000')
    # plt.plot([-500 * 0.31490776 / 0.39821976, 500 * 0.31490776 / 0.39821976], [500, -500], c='#007F7F', linewidth=0.5)
    # plt.plot([-500 * 0.3650696 / 0.45955154, 500 * 0.3650696 / 0.45955154], [500, -500], c='#007F7F', linewidth=0.5)
    # plt.show()
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_3.jpeg', dpi=2000)
    # # input-output graph ##############################################################################



    # # \bar\phi(18) 18 ##############################################################################
    print(w1[18])
    s = [-w1[18, 1] * 100, w1[18, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    # print(v1)
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[18] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3[0])

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-3.9, 0.9))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    # 设置坐标轴
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([-3.5, -1.5, 0.5])
    plt.xticks(x_tick)
    plt.yticks(y_tick)

    plt.plot(index, np.array(v3_record)*5, linewidth=0.5, color='#007F00')
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_4.jpeg', dpi=2000)
    # # \bar\phi(18) 18 ##############################################################################
    #
    # # \bar\phi(15) 15 ##############################################################################
    print(w1[15])
    s = [-w1[15, 1] * 100, w1[15, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[15] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3[0])

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-0.3, 3.3))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([0, 1.5, 3])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.plot(index, np.array(v3_record) * 5, linewidth=0.5, color='#007F00')
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_5.jpeg', dpi=2000)
    # # \bar\phi(15) 15 ##############################################################################

    # # \bar\phi(26) 26 ##############################################################################
    print(w1[26])
    s = [-w1[26, 1] * 100, w1[26, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[26] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3[0])

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-4.72, -3.28))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([-4.6, -4, -3.4])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.plot(index, np.array(v3_record) * 5, linewidth=0.5, color='#007F00')
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_6.jpeg', dpi=2000)
    # # \bar\phi(26) 26 ##############################################################################

    # # \bar\phi(28) 28 ##############################################################################
    print(w1[28])
    s = [w1[28, 1] * 100, -w1[28, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[28] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3[0])

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(2.9, 4.1))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    # 设置坐标轴
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([3, 3.5, 4])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.plot(index, np.array(v3_record) * 5, linewidth=0.5, color='#007F00')
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_7.jpeg', dpi=2000)
    # # \bar\phi(26) 26 ##############################################################################

    # # \bar\phi(14) 14 ##############################################################################
    print(w1[14])
    s = [-w1[14, 1] * 100, w1[14, 0] * 100]
    v1 = np.tanh(np.matmul(w1, s))
    # print(v1)
    feature = v1
    feature[np.argwhere(feature < 0)] = -1
    feature[np.argwhere(feature > 0)] = 1
    index = np.linspace(-1, 1, 100)
    v3_record = []
    for i in index:
        feature[14] = i

        v2 = np.tanh(np.matmul(w2, feature))
        v3 = np.tanh(np.matmul(w3, v2))
        v3_record.append(v3[0])

    fig = plt.figure(figsize=[4 / 2.54, 3.5 / 2.54])
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(3.92, 4.88))
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95, wspace=None, hspace=0.5)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(8)
    x_tick = np.array([-1, 0, 1])
    y_tick = np.array([4, 4.4, 4.8])
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.plot(index, np.array(v3_record) * 5, linewidth=0.5, color='#007F00')
    plt.grid(color='#999999', ls='--', lw=0.25)
    fig.savefig('C:/Users/asus/Desktop/PPO_8.jpeg', dpi=2000)
    # # \bar\phi(14) 14 ##############################################################################


if __name__ == '__main__':
    main(0.9, 8, 146)

