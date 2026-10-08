"""Phase 3 analysis: locate the liquid-gas critical point of the 2D LJ fluid.

Reads the chunks written by run_critical_scan.py and produces

  1. an equilibration check (segment means of E/N must not trend),
  2. Binder cumulants U_L(T) of the sub-box density for several sub-box sizes
     L_b; for a 2D-Ising critical point the curves cross at T_c with
     U* ~ 0.61 (subsystem-block method, Rovere, Heermann & Binder 1990),
  3. coexistence densities rho_gas / rho_liq from the two peaks of the sub-box
     density distribution below T_c, fitted with the Ising exponent beta = 1/8
     and the law of rectilinear diameters,
  4. the configurational heat capacity C_v/N = N var(E/N) / T^2 with blocking
     errors, computed from equilibrated data only,
  5. a diagnostic of the original Phase 3 data (fss_final_N*.npz).

Usage:
  python phase3/analyze_critical.py /tmp/lj_scan/N16384 --out phase3/results
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lj2d import binder_cumulant, blocking_error, jackknife  # noqa: E402

BETA_ISING = 1.0 / 8.0
U_STAR_ISING = 0.6107  # critical Binder cumulant, 2D Ising, periodic square (Kamieniarz & Bloete 1993)
HERE = os.path.dirname(os.path.abspath(__file__))


def load_scan(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    chunks = [np.load(f) for f in sorted(glob.glob(os.path.join(path, "chunk_*.npz")))]
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].files}
    return cfg, data


# ---------------------------------------------------------------- crossings
def crossings(T, U_small, U_large):
    """Temperatures where U_large - U_small changes sign (linear interpolation)."""
    d = U_large - U_small
    out = []
    for i in range(len(T) - 1):
        if d[i] * d[i + 1] < 0:
            t = T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i])
            u = np.interp(t, T[i:i + 2], U_small[i:i + 2])
            out.append((t, u))
    return out


# --------------------------------------------------------- coexistence peaks
def two_peaks(counts, area, smooth_rho=0.012):
    """Low- and high-density maxima of the sub-box density distribution.

    Histogrammed on integer particle counts (density is quantised in steps of
    1/area, so arbitrary bins alias), then smoothed with a Gaussian of width
    smooth_rho in density units.
    """
    h = np.bincount(np.asarray(counts, dtype=np.int64).ravel()).astype(float)
    h = gaussian_filter1d(h, smooth_rho * area)
    c = np.arange(len(h)) / area
    peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] > 0.05 * h.max()]
    if len(peaks) < 2:
        return None
    return c[peaks[0]], c[peaks[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="directory written by run_critical_scan.py")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--discard-steps", type=int, default=None, help="default: the scan's equil-steps")
    ap.add_argument("--coex-n", type=int, default=12, help="sub-box grid used for coexistence densities")
    ap.add_argument("--min-sub", type=int, default=4,
                    help="exclude sub-box grids coarser than this (blocks too close to the full box)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg, d = load_scan(a.scan)
    T = np.array(cfg["temps"])
    L, N, rho = cfg["L"], cfg["N"], cfg["rho"]
    discard = cfg["equil_steps"] if a.discard_steps is None else a.discard_steps
    keep_e = d["step"] > discard
    keep_s = d["s_step"] > discard
    E = d["E"][keep_e]                   # [frames, B]
    subdivs = sorted(int(k.split("_")[1]) for k in d if k.startswith("counts_"))
    subdivs = [n for n in subdivs if n >= a.min_sub]
    n_frames = int(keep_s.sum())
    print(f"N={N} rho={rho} L={L:.1f}; {n_frames} configurations per T after discarding {discard} steps")
    summary = {"N": N, "rho": rho, "L": L, "temps": T.tolist(), "discard_steps": discard, "frames": n_frames}

    # 1. Equilibration check -------------------------------------------------
    seg = np.array([[blocking_error(s, 5) for s in np.array_split(E[:, b], 4)] for b in range(len(T))])
    drift = seg[:, -1, 0] - seg[:, 0, 0]
    drift_err = np.hypot(seg[:, -1, 1], seg[:, 0, 1])
    summary["E_drift_first_to_last_quarter"] = drift.tolist()
    summary["E_drift_err"] = drift_err.tolist()

    # 2. Binder cumulants -----------------------------------------------------
    U, U_err, Lb = {}, {}, {}
    for n in subdivs:
        c = d[f"counts_{n}"][keep_s].astype(np.float64)   # [frames, B, n*n]
        Lb[n] = L / n
        rho_b = c / Lb[n] ** 2
        vals = []
        for b in range(len(T)):
            # Each frame contributes n*n sub-box samples; jackknife over time blocks.
            u, err = jackknife(lambda x: binder_cumulant(x.ravel()), rho_b[:, b, :], n_blocks=10)
            vals.append((u, err))
        U[n], U_err[n] = np.array(vals).T
    cross = []
    sizes = sorted(subdivs, reverse=True)  # increasing L_b
    for small, large in zip(sizes[:-1], sizes[1:]):
        for t, u in crossings(T, U[small], U[large]):
            cross.append({"Lb_small": Lb[small], "Lb_large": Lb[large], "T": t, "U": u})
    summary["binder"] = {f"Lb={Lb[n]:.2f}": {"U": U[n].tolist(), "err": U_err[n].tolist()} for n in subdivs}
    summary["binder_crossings"] = cross

    # 3. Coexistence densities -------------------------------------------------
    nco = a.coex_n
    area_co = (L / nco) ** 2
    counts_co = d[f"counts_{nco}"][keep_s]
    rho_co = counts_co / area_co
    coex = []
    for b, t in enumerate(T):
        pk = two_peaks(counts_co[:, b, :], area_co)
        if pk:
            coex.append((t, *pk))
    coex = np.array(coex)
    fit = None
    if len(coex) >= 3:
        def order(t, A, Tc):
            return A * np.clip(Tc - t, 0, None) ** BETA_ISING
        try:
            (A, Tc_fit), cov = curve_fit(order, coex[:, 0], coex[:, 2] - coex[:, 1], p0=(1.0, T.max()))
            diam = 0.5 * (coex[:, 1] + coex[:, 2])
            slope, rho_c = np.polyfit(coex[:, 0] - Tc_fit, diam, 1)
            fit = {"Tc": Tc_fit, "Tc_err": float(np.sqrt(cov[1, 1])), "A": A, "rho_c": rho_c, "diam_slope": slope}
        except RuntimeError:
            fit = None
    summary["coexistence"] = coex.tolist()
    summary["coexistence_fit_beta_1_8"] = fit

    # 4. Heat capacity ---------------------------------------------------------
    cv = []
    for b, t in enumerate(T):
        val, err = jackknife(lambda x: N * np.var(x) / t ** 2, E[:, b], n_blocks=10)
        cv.append((val, err))
    cv = np.array(cv)
    summary["Cv_per_particle"] = cv[:, 0].tolist()
    summary["Cv_err"] = cv[:, 1].tolist()

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    np.savez_compressed(os.path.join(a.out, "binder_summary.npz"), T=T,
                        Lb=np.array([Lb[n] for n in subdivs]),
                        U=np.array([U[n] for n in subdivs]), U_err=np.array([U_err[n] for n in subdivs]),
                        Cv=cv[:, 0], Cv_err=cv[:, 1], coex=coex)

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(subdivs)))
    for (n, col) in zip(sorted(subdivs, reverse=True), colors):
        ax[0, 0].errorbar(T, U[n], U_err[n], marker="o", ms=4, capsize=2, color=col, label=f"$L_b$ = {Lb[n]:.1f}")
    ax[0, 0].axhline(U_STAR_ISING, color="k", ls=":", label="2D Ising $U^*$")
    for c in cross:
        ax[0, 0].plot(c["T"], c["U"], "rx", ms=8)
    ax[0, 0].set_xlabel("T")
    ax[0, 0].set_ylabel("Binder cumulant $U_L$ of sub-box density")
    ax[0, 0].set_title("Sub-box Binder cumulants (crossings marked x)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.4)

    tsel = np.linspace(0, len(T) - 1, min(len(T), 5)).astype(int)
    for b, col in zip(tsel, plt.cm.coolwarm(np.linspace(0, 1, len(tsel)))):
        h = np.bincount(counts_co[:, b, :].ravel()).astype(float)
        h = gaussian_filter1d(h / h.sum() * area_co, 0.006 * area_co)
        ax[0, 1].plot(np.arange(len(h)) / area_co, h, lw=1.6, color=col, label=f"T = {T[b]:.2f}")
    ax[0, 1].set_xlabel(r"sub-box density $\rho_b$" + f"  ($L_b$ = {L / nco:.1f})")
    ax[0, 1].set_ylabel(r"$P(\rho_b)$")
    ax[0, 1].set_title("Sub-box density distribution")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.4)

    if len(coex):
        ax[1, 0].plot(coex[:, 1], coex[:, 0], "bo", label=r"$\rho_{gas}$")
        ax[1, 0].plot(coex[:, 2], coex[:, 0], "ro", label=r"$\rho_{liq}$")
        if fit:
            tt = np.linspace(coex[:, 0].min(), fit["Tc"], 200)
            half = 0.5 * fit["A"] * (fit["Tc"] - tt) ** BETA_ISING
            mid = fit["rho_c"] + fit["diam_slope"] * (tt - fit["Tc"])
            ax[1, 0].plot(mid - half, tt, "k-", lw=1)
            ax[1, 0].plot(mid + half, tt, "k-", lw=1, label=r"fit, $\beta = 1/8$")
            ax[1, 0].plot(fit["rho_c"], fit["Tc"], "k*", ms=14,
                          label=f"$T_c$={fit['Tc']:.3f}, $\\rho_c$={fit['rho_c']:.3f}")
    ax[1, 0].set_xlabel(r"$\rho$")
    ax[1, 0].set_ylabel("T")
    ax[1, 0].set_title("Coexistence curve from sub-box density peaks")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.4)

    ax[1, 1].errorbar(T, cv[:, 0], cv[:, 1], marker="o", capsize=3, color="darkgreen")
    ax[1, 1].set_xlabel("T")
    ax[1, 1].set_ylabel(r"$C_v^{conf}/N = N\,\mathrm{var}(E/N)/T^2$")
    ax[1, 1].set_title(f"Configurational heat capacity (N={N:,}, equilibrated)")
    ax[1, 1].grid(alpha=0.4)
    fig.suptitle(f"2D Lennard-Jones critical point, N={N:,}, $\\rho$={rho}, $r_c$=2.5", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "critical_point_analysis.png"), dpi=150)
    plt.close(fig)

    # Equilibration figure
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = d["step"]
    for b, col in enumerate(plt.cm.coolwarm(np.linspace(0, 1, len(T)))):
        ax.plot(steps * cfg["dt"], d["E"][:, b], color=col, lw=0.6, label=f"T={T[b]:.2f}")
    ax.axvline(discard * cfg["dt"], color="k", ls="--", label="end of equilibration")
    ax.set_xlabel("time (LJ units)")
    ax.set_ylabel("E/N")
    ax.set_title("Potential energy time series: data before the dashed line is discarded")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "equilibration.png"), dpi=150)
    plt.close(fig)

    print(json.dumps({"binder_crossings": cross, "coexistence_fit": fit,
                      "max_|drift|/err": float(np.max(np.abs(drift) / drift_err))}, indent=2, default=float))


if __name__ == "__main__":
    main()
