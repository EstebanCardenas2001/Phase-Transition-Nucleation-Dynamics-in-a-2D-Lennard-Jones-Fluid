"""Diagnostic of the original Phase 3 finite-size-scaling data (fss_final_N*.npz).

The original analysis (v1_original/data_colapse.py) computed C_v = N var(E/N) / T^2 and
read C_v ~ N as the signature of a first-order transition. This script shows
that the variance is dominated by a slow downward drift of E/N during the
measurement window, i.e. the systems were still condensing:

  * left panel: E/N(t) for every N at one temperature, with the drift visible;
  * right panel: the total standard deviation of E/N (flat in N, as a drift of
    the same size per particle would give) against the short-time standard
    deviation estimated from successive differences (falls as N^-1/2, as
    equilibrium fluctuations should).

Because N var(E/N) multiplies an N-independent drift by N, C_v grows like N
whatever the order of the transition.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sizes = [1024, 4096, 16384]
    data = {n: np.load(os.path.join(HERE, "v1_original", f"fss_final_N{n}.npz")) for n in sizes}
    T = data[sizes[0]]["T_target"]
    k = int(np.argmin(np.abs(T - 0.335)))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    rows = []
    for n, col in zip(sizes, colors):
        E = data[n]["E_series"]
        ax[0].plot(E[:, k], color=col, lw=0.5, alpha=0.8, label=f"N={n}")
        quarters = E[: len(E) // 4 * 4].reshape(4, -1, E.shape[1]).mean(1)
        total_sd = E.std(0)
        short_sd = np.diff(E, axis=0).std(0) / np.sqrt(2)
        rows.append((n, total_sd.mean(), short_sd.mean(), (quarters[-1] - quarters[0]).mean()))
    ax[0].set_xlabel("sample index")
    ax[0].set_ylabel("E/N")
    ax[0].set_title(f"Original data, T = {T[k]:.3f}: E/N is still drifting")
    ax[0].legend()
    ax[0].grid(alpha=0.4)

    rows = np.array(rows)
    ax[1].loglog(rows[:, 0], rows[:, 1], "o-", label="total std of E/N (used for $C_v$)")
    ax[1].loglog(rows[:, 0], rows[:, 2], "s-", label="short-time std of E/N")
    ref = rows[0, 2] * np.sqrt(rows[0, 0] / rows[:, 0])
    ax[1].loglog(rows[:, 0], ref, "k:", label=r"$\propto N^{-1/2}$ (equilibrium)")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("std(E/N), averaged over T")
    ax[1].set_title("Variance is set by drift, not by fluctuations")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.4, which="both")
    fig.tight_layout()
    out = os.path.join(HERE, "results", "original_fss_diagnostic.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("N, total std, short-time std, mean drift (last - first quarter)")
    for r in rows:
        print(f"{int(r[0]):6d}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:+.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
