from collections import defaultdict
import math

import matplotlib.pyplot as plt
import numpy as np


class EnvAsym(object):
    def __init__(self, H=6, s_size=6, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.s_size = s_size
        self.H = H
        self.A = list(range(self.s_size))
        self._a_trans = self._gen_a_trans()
        self._R = self._gen_r()
        self._s_coeff = 1.0
        self._variance = 3  # You can tune this
        self._num_samples = 1000

    def _kernel(self, z1, z2):
        normalizing_const = math.sqrt(math.pi / self._variance)
        return math.exp(- self._variance * ((z1[0] - z2[0]) ** 2 + (z1[1] - z2[1]) ** 2)) / normalizing_const

    def _gen_r(self):
        #R =list(range(self.s_size))
        R = []
        for i in range(self.s_size) :
            R.append(self.rng.choice(list(range(self.s_size))))
        return R

    def _get_random_states(self):
        line_space = np.linspace(0, self.s_size, self._num_samples)
        return round(self.rng.choice(line_space),2)

    def _gaussian_sampler(self, mu, var=1.0 ):
        line_space = np.linspace(mu-0.5*self.s_size, mu+0.5*self.s_size, self._num_samples)
        p = [math.exp(- self._variance * (x - mu ) ** 2) for x in line_space ]
        p = [x/sum(p) for x in p]
        sample = self.rng.choice(line_space,p=p)
        return sample % self.s_size

    def _gen_a_trans(self):
        return self.rng.permutation(list(range(self.s_size)))

    def _get_sn(self, s, a):
        mu = (self._a_trans[a] + self._s_coeff*s) % self.s_size
        return round(self._gaussian_sampler(mu, var=self._variance),2)

    def _get_r(self, s, a):
        return round(math.exp(- 0.2 * (s - math.ceil(self.s_size/2) ) ** 2),3)

    def reset_rng(self, seed=0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def gen_init_states(self):
        return self._get_random_states()

    def random_pi(self):
        return self.rng.choice(range(self.s_size))

    def get_r_sn(self, s, a):
        r = self._get_r(s, a)
        sn = self._get_sn(s, a)
        return r, sn

    def gen_random_trajs(self, N, length, labeled):
        trajs = defaultdict(list)
        for _ in range(N):
            s = self.gen_init_states()
            for i in range(length):
                a = self.random_pi()
                r, sn = self.get_r_sn(s, a)
                if labeled:
                    trajs[i].append((s, a, r))
                else:
                    trajs[i].append((s, a))
                s = sn
        return trajs

    def gen_dataset(self, N1=15, N2=10, H=3):
        self.reset_rng(self.seed)
        D1 = self.gen_random_trajs(N=N1, length=H, labeled = True)
        D2 = self.gen_random_trajs(N=N2, length=H, labeled = False)
        return D1,D2


def visualize_distribution(s_primes, fname):
    plt.clf()
    plt.hist(s_primes, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # Add titles and labels
    plt.title('Distribution of s_primes')
    plt.xlabel('s_prime values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.savefig(fname)

def debug_trans():
    env = EnvAsym()
    for s in range(env.s_size):
        print(f"s0={s}")
        for a in range(env.s_size):
            print(f"s,a={s,a}")
            s_primes=[]
            for p in range(200):
                s_prime = env._get_sn(s, a)
                s_primes.append(s_prime)
            visualize_distribution(s_primes, fname=f"results/fig_s_{s}_t_|{'|'.join([str(int(x + s * env._s_coeff) % env.s_size) for x in env.a_trans])}|_a_{a}.png")
            print(f"s1={s}")

if __name__ == '__main__':
    env = EnvAsym()
    results = env.gen_dataset()
    print(1)
