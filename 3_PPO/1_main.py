from Agent import PPO
from Env import Env
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def main(gamma, index):
    # start = time.time()
    # start_time = time.time()
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/PPO_'+str(gamma)+'/'+str(index)
    if not os.path.exists(path):
        os.makedirs(path)
    agent = PPO(path, s_dim=2, gamma=gamma)
    env = Env()

    for episode in range(200):
        s = env.reset()
        init_error = s[0]
        for ep_step in range(500):
            a = agent.get_action(s)
            s_, r = env.step(a[0])
            agent.store_transition(s, a, s_, r)
            s = s_
        last_error = s[0]
        print('episode: ', episode,
              ' init_error: ', round(init_error, 3),
              ' last_error: ', round(last_error, 5),
              )
        agent.store_net(str(episode))


if __name__ == '__main__':
    for i in range(10):
        main(0.9, i)