import json

import matplotlib.pyplot as plt
import numpy as np

colors = [
    (1, 0, 0),
    (0.7, 0.4, 0),
    (0.7, 0.7, 0),
    (0.4, 0.7, 0),
    (0, 1, 0),
    (0, 0.7, 0.7),
    (0, 0, 1),
]


def linear_regression(n1s, n2s):
    lamda = 0
    X = []
    Y = []
    r_exp = []
    for i, n1 in enumerate(n1s):
        jitems = json.load(open(f"results/env_kernel_n1_{n1}.json"))
        rdict = dict((n2, r) for n2, r in zip(jitems['n2'], jitems['r1']))
        for n2 in n2s:
            if n2 in rdict:
                r = rdict[n2]
                X.append((n1 ** (-0.5), n2 ** (-0.5), 1))
                Y.append(r)
                r_exp.append(((n1, n2), r))
    r_exp = dict(r_exp)
    X = np.array(X)
    Y = np.array(Y)
    w = np.dot(np.matmul(np.linalg.inv(np.matmul(X.T, X) + lamda * np.eye(X.shape[1])), X.T), Y)
    r_theory = []
    for n1 in n1s:
        for n2 in n2s:
            x = np.array([n1 ** (-0.5), n2 ** (-0.5), 1])
            r = np.dot(w, x)
            r_theory.append(((n1, n2), r))
    r_theory = dict(r_theory)
    return r_exp, r_theory


def visualize_distribution_exp():
    plt.clf()
    plt.figure(figsize=(10, 6))
    n1s = [10, 20, 50, 100, 200, 500]
    for i, n1 in enumerate(n1s):
        jitems = json.load(open(f"results/env_kernel_n1_{n1}.json"))
        plt.plot(jitems['n2'], jitems['r1'], label=f'N_1={n1}', color=colors[i])
        # if n == n1s[0]:
        #    plt.axhline(y=jitems['rrand'], color='r', linestyle='--', label=f'random = {r_rand}')
    plt.xlabel('$N_2$')
    plt.ylabel('$V_1^{\pi}(s)$')
    plt.title('Plot of $V_1^{\pi}(s)$ vs $N_2$ and $N_1$')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/out_exp.png")


def regression_theory():
    n1s = [10, 20, 50, 100]
    n2s = [10, 20, 50, 100, 200, 500]
    r_exp, r_theory = linear_regression(n1s, n2s)
    visualize_distribution_theory(n1s, n2s, r_exp, "exp")
    visualize_distribution_theory(n1s, n2s, r_theory, "theory")

    visualize_distribution_theory_tikz(n1s, n2s, r_exp, "exp")
    visualize_distribution_theory_tikz(n1s, n2s, r_theory, "theory")


def visualize_distribution_theory_tikz(n1s, n2s, r_vals, fname):
    f = open(f"results/plot_{fname}.tex", "w")
    fdict = {
        "theory": "Theoretical",
        "exp": "Experimental",
    }
    f.write("""
\\begin{tikzpicture}
    \\begin{axis}[
    width=8cm,
    height=5cm,
    legend pos=south east,
    grid=major,
    grid style={dashed,gray!30},
    xmin=10, xmax=500,
    ymin=4, ymax=6.5,
    xlabel={$N_2$},
    ylabel={$V_1^{\\pi}(s)$},
    font=\\scriptsize,
    title={WORD1 Values of $V_1^{\\pi}(s)$},
    xlabel style={
        at={(current axis.south east)}, % Relative positioning
        anchor=north east,              % Anchoring at a specific point
        yshift=5pt,                   % Shifting downward
        xshift=10pt                      % Shifting rightward (if needed)
    },
    ylabel style={
        at={(current axis.north west)}, % Relative positioning
        anchor=north east,              % Anchoring at a specific point
        yshift=-10pt,                   % Shifting downward
        xshift=20pt                      % Shifting rightward (if needed)
    },
]
""".replace("WORD1", fdict[fname]))
    for i, n1 in enumerate(n1s):
        f.write("\\addplot[c%s, thick] table[row sep=\\\\]{\nx y \\\\ \n" % (i + 1))
        for n2 in n2s:
            print(n1, n2)
            if (n1, n2) in r_vals.keys():
                # r1e.append(r_vals[(n1,n2)])
                # n2s_plot.append(n2)
                # plt.plot(n2s_plot, r1e, label=f'$N_1={n1}$', color=colors[i])
                r1 = r_vals[(n1, n2)]
                f.write(f" {n2}  {r1} \\\\ \n")
        f.write("};\n")
        f.write("\\addlegendentry{$N_1=%s$}\n" % n1)
    f.write("""\\end{axis}
\\end{tikzpicture}
""")


def visualize_distribution_theory(n1s, n2s, r_vals, fname):
    plt.clf()
    plt.figure(figsize=(10, 6))
    for i, n1 in enumerate(n1s):
        r1e = []
        n2s_plot = []
        for n2 in n2s:
            print(n1, n2)
            if (n1, n2) in r_vals.keys():
                r1e.append(r_vals[(n1, n2)])
                n2s_plot.append(n2)
        plt.plot(n2s_plot, r1e, label=f'$N_1={n1}$', color=colors[i])
        # if n == n1s[0]:
        #    plt.axhline(y=jitems['rrand'], color='r', linestyle='--', label=f'random = {r_rand}')
    plt.xlabel('$N_2$')
    plt.ylabel('$V_1^{\pi}(s)$')
    plt.title('Plot of $V_1^{\pi}(s)$ vs $N_2$ and $N_1$')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/out_{fname}.png")


if __name__ == '__main__':
    regression_theory()
