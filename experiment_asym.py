import json
import os

import numpy as np

from environment_asym import EnvAsym
from pds_kernel import PDSKernel, kernel_gaussian, phi_tuple


def evaluate(env, pi_func):
    r1s = []
    for i in range(2):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn = env.get_r_sn(sn, a)
            r1 += r
        r1s.append(r1)
    return np.average(r1s)

def env_experiments(env, pds, n1, n2, H):
    r1s = []
    for i in range(5):
        env.reset_rng(i)
        D1, D2 = env.gen_dataset(N1=n1, N2=n2, H=H)
        pi_hat, pi_bandit_hat= pds.data_sharing_kernel_approx(D1, D2)
        r1 = evaluate(env=env, pi_func=pi_hat)
        r1s.append(r1)
    r1 = np.average(r1s)
    print(f"N1={n1}, N2={n2}, R1={r1}")
    return r1

def save_result(n2s, r1s, r_rand, fname):
    json.dump({'n2':n2s, 'r1':r1s, 'rrand':r_rand},open(fname.replace(".png",".json"),"w"), indent=2 )


def run():
    H = 8
    s_size = 8
    envs = {
            'kernel' : EnvAsym(s_size=s_size, H=H),
            }


    n1s = [10,20,50,100,200]
    n2s = [10,20,50,100,200,500]

    os.makedirs("results", exist_ok=True)
    for ekey, env in envs.items():
        pds = PDSKernel(env=env, kernel=kernel_gaussian, phi=phi_tuple)

        for n1 in n1s: #,100]:
            def random_pi(h, s):
                return env.random_pi()
            r_rand = evaluate(env=env, pi_func=random_pi)
            print(f"r_rand={r_rand}")
            r1s = []
            for n2 in n2s:
                r1 = env_experiments(env, pds, n1, n2, H)
                r1s.append(r1)
            fname = f"results/env_{ekey}_n1_{n1}.png"
            save_result(n2s, r1s, r_rand, fname)

if __name__ == "__main__":
    run()
