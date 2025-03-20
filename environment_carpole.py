from collections import defaultdict

import gym
import numpy as np


class EnvCarpole(object):
    def __init__(self, H=60, seed=0):
        self.env = gym.make("CartPole-v1")
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.H = H
        self.A = list(range(2))

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        return self.env.reset(seed=self.seed)[0]

    def random_pi(self):
        return self.rng.choice(self.A)

    def get_r_sn(self, s, a):
        sn, r, done, _, _ = self.env.step(a)
        return r, sn, done

    def gen_random_trajs(self, N, length, labeled):
        trajs_list = []
        for _ in range(N):
            traj = []
            s = self.gen_init_states()
            for i in range(length):
                a = self.random_pi()
                r, sn, done = self.get_r_sn(s, a)
                if labeled:
                    traj.append((s, a, r))
                else:
                    traj.append((s, a))
                s = sn
                if done:
                    break
            trajs_list.append(traj)
        trajs_list = sorted(trajs_list, key=lambda x: len(x), reverse=True)
        trajs = defaultdict(list)
        for i in range(length):
            for traj in trajs_list:
                if i < len(traj):
                    trajs[i].append(traj[i])
        return trajs

    def gen_dataset(self, N1=15, N2=10, H=30):
        self.reset_rng(self.seed)
        D1 = self.gen_random_trajs(N=N1, length=H, labeled = True)
        D2 = self.gen_random_trajs(N=N2, length=H, labeled = False)
        self.H = len(D1.keys())
        return D1,D2


if __name__ == '__main__':
    env = EnvCarpole()
    results = env.gen_dataset(N1=50, N2=100, H=100)
    print(1)
