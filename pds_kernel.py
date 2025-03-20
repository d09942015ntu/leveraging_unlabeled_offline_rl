import math

import numpy as np


class RewardEval(object):
    def __init__(self, phi, kernel, Zh, lambda_inv, alpha, lamda, Aspace, B, h, H):
        self.Zh = Zh
        self.kernel = kernel
        self.lambda_inv = lambda_inv
        self.alpha = alpha
        self.lamda = lamda
        self.phi = phi
        self.B = B
        self.h = h
        self.H = H
        self.Aspace = Aspace

    def mean_kernel_sample(self, z):
        k_array = np.array([self.kernel(z, zi) for zi in self.Zh])
        result = k_array.dot(self.alpha)
        return result

    def var_kernel_sample(self, z):
        k_array = np.array([self.kernel(z, zi) for zi in self.Zh])
        uncertainty = max(self.kernel(z, z) - np.dot(np.dot(self.lambda_inv, k_array), k_array), 0)
        result = (self.lamda ** 0.5) * ((uncertainty) ** 0.5)
        return result

    def Zhat_h_func(self, z):
        return max(self.mean_kernel_sample(z) - self.B * self.var_kernel_sample(z), 0)

    def Qhat_h_func(self, s, a):
        z = self.phi(s, a)
        result = np.clip(self.Zhat_h_func(z), 0, self.H - self.h)
        return result

    def Vhat_h_func(self, s):
        max_q, _ = max([(self.Qhat_h_func(s, a), a) for a in self.Aspace], key=lambda x: x[0])
        return max_q


class PDSKernel(object):
    def __init__(self, env, phi, kernel, beta1=0.05, beta2=0.05, lamda1=1, lamda2=1):
        self.env = env
        self.phi = phi
        self.kernel = kernel
        self.H = env.H
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        pass

    def build_kernel_matrix(self, Zh, lamda, Rh):
        N1 = len(Zh)
        # Build kernel matrix K of size NxN
        K = np.zeros((N1, N1))
        for i in range(N1):
            zi = Zh[i]  # combine s,a if needed
            for j in range(N1):
                zj = Zh[j]
                # print(f"zi,zj={zi},{zj}")
                K[i, j] = self.kernel(zi, zj)

        lambda_inv = np.linalg.inv(K + lamda * np.eye(len(Zh)))
        # try:
        alpha = lambda_inv.dot(Rh)
        # except:
        #    print(1)
        return K, lambda_inv, alpha

    def data_preprocessing(self, D, h):
        Sh = []
        Ah = []
        Rh = []
        for (s_h_t, a_h_t, r_h_t) in D[h]:
            Sh.append(s_h_t)
            Ah.append(a_h_t)
            Rh.append(r_h_t)
        Sh = np.array(Sh)
        Ah = np.array(Ah)
        Rh = np.array(Rh)
        Zh = np.array([self.phi(s, a) for s, a in zip(Sh, Ah)])
        return Sh, Ah, Rh, Zh

    def fit_reward_function(self, D1):
        reward_fn = []

        for h in range(self.env.H):
            Sh, Ah, Rh, Zh = self.data_preprocessing(D1, h)

            K, lambda_inv, alpha = self.build_kernel_matrix(Zh, self.lamda1, Rh)

            lambda_inv = np.linalg.inv(K + self.lamda1 * np.eye(len(Sh)))
            alpha = lambda_inv.dot(Rh)

            reward_fn.append(
                RewardEval(self.phi, self.kernel, Zh, lambda_inv, alpha, self.lamda1, self.env.A, self.beta1, h,
                           self.env.H))

        return reward_fn

    def relabel_unlabeled_data(self, D1, D2, reward_fn, relabel_D1=True):

        Dtilde = []
        for h in range(self.env.H):
            Dtilde.append([])
            reward_fn_h = reward_fn[h]

            for (s_h_t, a_h_t, r_pess) in D1[h]:
                if relabel_D1:
                    r_pess = reward_fn_h.Zhat_h_func(self.phi(s_h_t, a_h_t))
                Dtilde[-1].append((s_h_t, a_h_t, r_pess))

            for (s_h_t, a_h_t) in D2[h]:
                r_pess = reward_fn_h.Zhat_h_func(self.phi(s_h_t, a_h_t))
                Dtilde[-1].append((s_h_t, a_h_t, r_pess))

        return Dtilde

    def pevi_kernel_approx(self, Dtheta):

        rl_fn = []

        Sh1 = []
        for h in reversed(range(self.env.H)):

            Sh, Ah, Rh, Zh = self.data_preprocessing(Dtheta, h)

            if len(rl_fn) > 0:
                Rh_p_V = []
                for r in Rh:
                    Rh_p_V.append(r)
                for i, s in enumerate(Sh1):
                    Rh_p_V[i] += rl_fn[0].Vhat_h_func(s)
                # Rh_p_V = [r + rl_fn[0].Vhat_h_func(s) for r,s in zip(Rh, Sh1)]
            else:
                Rh_p_V = Rh

            K, lambda_inv, alpha = self.build_kernel_matrix(Zh, self.lamda2, Rh_p_V)

            rewad_eval = RewardEval(self.phi, self.kernel, Zh, lambda_inv, alpha, self.lamda2, self.env.A, self.beta2,
                                    h, self.env.H)

            rl_fn.insert(0, rewad_eval)
            Sh1 = Sh.tolist()

        return rl_fn

    def data_sharing_kernel_approx(self, D1, D2):
        # 1) Learn the reward function \hat{\theta}_h
        reward_fn = self.fit_reward_function(D1)

        ## 2) Relabel unlabeled data D2 with tilde{theta}
        Dtheta = self.relabel_unlabeled_data(D1, D2, reward_fn)

        ## 3) Learn the policy from the relabeled dataset using PEVI (Algorithm 2)
        rl_fn = self.pevi_kernel_approx(Dtheta)

        def pi_reward_hat(h, s):
            _, max_a = max([(reward_fn[h].Qhat_h_func(s, a), a) for a in self.env.A], key=lambda x: x[0])
            return max_a

        def pi_rl_fn(h, s):
            _, max_a = max([(rl_fn[h].Qhat_h_func(s, a), a) for a in self.env.A], key=lambda x: x[0])
            return max_a

        return pi_rl_fn, pi_reward_hat


def phi_tuple(s, a):
    z = (s, a)
    return z


def phi_array(s, a):
    s = list(s)
    z = s + [a]
    z = tuple(z)
    return z

def phi_array_64_4(s, a):
    z = tuple((s, a))
    return z


def phi_tabular_64_4(s, a):
    # print(f"s={s}, a={a}")
    z = np.zeros((64, 4))
    z[s, a] = 1
    z = z.flatten()
    return z


def phi_quadratic_1(s, a):
    s = list(s)
    z1 = np.array(s + [a, 1])
    z2 = tuple(np.matmul(z1[:, np.newaxis], z1[np.newaxis, :]).flatten())
    return z2


def phi_linear_2(s, a):
    s = np.array(s)
    a = np.array([a == 0, a == 1]).astype(float)
    z = tuple(np.concatenate([s, a, [1]]).flatten())
    return z


def phi_quadratic_2(s, a):
    s = list(s)
    a = np.array([a == 0, a == 1]).astype(float)
    z1 = np.concatenate([s, a, [1]])
    z2 = np.matmul(z1[:, np.newaxis], z1[np.newaxis, :])
    z2 = tuple(np.triu(z2).flatten())
    return z2


def phi_cubic_2(s, a):
    s = list(s)
    a = np.array([a == 0, a == 1]).astype(float)
    z1 = np.concatenate([s, a, [1]])
    Z2 = np.matmul(z1[:, np.newaxis], z1[np.newaxis, :])
    z2 = np.triu(Z2).flatten()
    Z3 = np.matmul(z2[:, np.newaxis], z1[np.newaxis, :])
    z3 = tuple(Z3.flatten())
    return z3


def phi_linear_3(s, a):
    s = np.array(s)
    a = np.array([a == 0, a == 1, a == 2]).astype(float)
    z = tuple(np.matmul(s[:, np.newaxis], a[np.newaxis, :]).flatten())
    z = np.concatenate([z, [1]])
    return z


def phi_quadratic_3(s, a):
    s = list(s)
    a = np.array([a == 0, a == 1, a == 2]).astype(float)
    z1 = np.concatenate([s, a, [1]])
    z2 = tuple(np.matmul(z1[:, np.newaxis], z1[np.newaxis, :]).flatten())
    return z2


def kernel_linear(z1, z2):
    return np.dot(z1, z2)


def kernel_gaussian(z1, z2, variance=3):
    # normalizing_const = math.sqrt(math.pi / variance)
    dist = 0
    for z1i, z2i in zip(z1, z2):
        dist += (z1i - z2i) ** 2
    return math.exp(- variance * dist)  # / normalizing_const


def evaluate(env, pi_func):
    R1 = []
    for i in range(100):
        env.reset_rng(i)
        r1 = 0
        sn = env.gen_init_states()
        for h in range(env.H):
            a = pi_func(h, sn)
            r, sn = env.get_r_sn(sn, a)
            r1 += r
        R1.append(r1)
    return np.average(R1)



if __name__ == "__main__":
    run_debug_linear()
    pass
