import os

import numpy as np
import json
import argparse

from environment_carpole import EnvCarpole
from pds_kernel import (PDSKernel, kernel_gaussian, kernel_linear, phi_array, phi_linear_2, phi_quadratic_2, phi_cubic_2)


def evaluate(env, pi_func, repeat):
    r1s = []
    for i in range(repeat):
        env.reset_rng(i)
        r1 = 0
        sn =env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn, done = env.get_r_sn(sn, a)
            r1 += r
            if done:
                break
        r1s.append(r1)
    return np.average(r1s)

def env_experiments(env, pds, n1, n2, H, output_dir, seed_start, repeat, repeat2):
    for i in range(seed_start,seed_start+repeat):
        env.reset_rng(i)
        D1, D2 = env.gen_dataset(N1=n1, N2=n2, H=H)
        pi_hat, pi_bandit_hat= pds.data_sharing_kernel_approx(D1, D2)
        r1 = evaluate(env=env, pi_func=pi_hat, repeat=repeat2)

        os.makedirs(output_dir,exist_ok=True)
        fname = f"{output_dir}/i_{i}_n1_{n1}_n2_{n2}.json"
        json_str = json.dumps({'i':i,'n1':n1, 'n2': n2, 'r1': r1})
        print(json_str)
        f = open(fname,"w")
        f.write(json_str)
        f.close()


def run(n1s,n2s, arg_kernel='kernel_gaussian',arg_phi='phi_array', output_dir="results", seed_start=0, repeat=1, repeat2=1):
    arg_str=f"n1s={n1s},n2s={n2s}, arg_kernel={arg_kernel},arg_phi={arg_phi}, output_dir={output_dir}, repeat={repeat}"
    print(arg_str)
    H = 100
    envs = {
            'carpole' : EnvCarpole(H=H),
            }

    if arg_kernel == 'kernel_gaussian':
        kernel = kernel_gaussian
    elif arg_kernel == 'kernel_linear':
        kernel = kernel_linear
    else:
        kernel = kernel_gaussian

    if arg_phi == 'phi_array':
        phi = phi_array
    elif arg_phi == 'phi_linear':
        phi = phi_linear_2
    elif arg_phi == 'phi_quadratic':
        phi = phi_quadratic_2
    elif arg_phi == 'phi_cubic':
        phi = phi_cubic_2
    else:
        phi = phi_array

    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir,"args.txt"),"w")
    f.write(arg_str)
    f.close()

    for ekey, env in envs.items():

        pds = PDSKernel(env=env, kernel=kernel, phi=phi)

        for n1 in n1s: #,100]:
            def random_pi(h, s):
                return env.random_pi()
            r_rand = evaluate(env=env, pi_func=random_pi, repeat=repeat2)
            print(f"r_rand={r_rand}")
            for n2 in n2s:
                env_experiments(env, pds, n1, n2, H, output_dir, seed_start, repeat, repeat2)





def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument('--kernel', type=str,  choices=['kernel_linear', 'kernel_gaussian'], default='kernel_linear')
    parser.add_argument('--phi', type=str,  choices=['phi_array', 'phi_linear','phi_quadratic','phi_cubic'], default='phi_cubic')
    parser.add_argument('--output_dir', type=str,  default='results')
    parser.add_argument('--repeat', type=int,  default=5)
    parser.add_argument('--repeat2', type=int,  default=20)
    parser.add_argument('--seed_start', type=int,  default=0)
    parser.add_argument('--n1s', type=int, nargs='+', default=[20,50,100,200])
    parser.add_argument('--n2s', type=int, nargs='+', default=[10,20,50,100,200,500])

    # Parse the arguments
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    current_pid = str(os.getpid())
    with open(os.path.join(args.output_dir,f"pid_{current_pid}.txt"), 'w') as f:
        f.write(str(os.getpid()))

    run(args.n1s, args.n2s, args.kernel, args.phi, args.output_dir, args.seed_start, args.repeat, args.repeat2)

if __name__ == "__main__":
    main()

