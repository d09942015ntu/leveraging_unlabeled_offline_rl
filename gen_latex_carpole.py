import glob
import json

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


def visualize_reward():
    n1s = [20, 50, 100, 200]
    n2s = [10, 20, 50, 100, 200, 500]
    kernels = [
        "kernel_gaussian",
        "kernel_cubic",
        "kernel_quadratic",
        "kernel_linear",
    ]

    for n1 in n1s:
        r_vals = []
        for k1 in kernels:
            for n2 in n2s:
                json_files = sorted(glob.glob(f"results/{k1}_{n1}/i_*_n1_{n1}_n2_{n2}.json"))
                r1s = [json.load(open(jfile, "r"))['r1'] for jfile in json_files]
                r1s = np.average(r1s)
                r_vals.append(((k1, n2), r1s))
        r_vals = dict(r_vals)
        visualize_distribution_theory_tikz(kernels, n2s, r_vals, n1)


def visualize_distribution_theory_tikz(kernels, n2s, r_vals, n1):
    f = open(f"results/plot_exp2_n1_{n1}.tex", "w")
    fdict = {
        20: "$N_1=20$",
        50: "$N_1=50$",
        100: "$N_1=100$",
        200: "$N_1=200$",
    }
    k1_vals = {
        "kernel_linear": "\\phi_{\\text{lin}}(z)",
        "kernel_quadratic": "\\phi_{\\text{quad}}(z)",
        "kernel_cubic": "\\phi_{\\text{cub}}(z)",
        "kernel_gaussian": "\\phi_{K}(z)",
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
    ymin=0, ymax=70,
    xlabel={$N_2$},
    ylabel={$V_1^{\\pi}(s)$},
    font=\\scriptsize,
    title={Values of $V_1^{\\pi}(s)$ when WORD1},
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
""".replace("WORD1", fdict[n1]))
    for i, k1 in enumerate(kernels):
        f.write("\\addplot[c%s, thick] table[row sep=\\\\]{\nx y \\\\ \n" % (i + 1))
        for n2 in n2s:
            print(k1, n2)
            if (k1, n2) in r_vals.keys():
                # r1e.append(r_vals[(n1,n2)])
                # n2s_plot.append(n2)
                # plt.plot(n2s_plot, r1e, label=f'$N_1={n1}$', color=colors[i])
                r1 = r_vals[(k1, n2)]
                f.write(f" {n2}  {r1} \\\\ \n")
        f.write("};\n")
        f.write("\\addlegendentry{$%s$}\n" % k1_vals[k1])
    f.write("""\\end{axis}
\\end{tikzpicture}
""")


if __name__ == '__main__':
    visualize_reward()
