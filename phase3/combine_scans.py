"""Combine the per-scan Phase 3 results into the final phase diagram.

Reads summary.json files written by analyze_critical.py and

  * fits the coexistence densities with the 2D-Ising scaling law
      rho_liq - rho_gas = A (T_c - T)^(1/8)
    and the law of rectilinear diameters
      (rho_liq + rho_gas) / 2 = rho_c + D (T - T_c),
  * estimates the fit's systematic error by leaving out one temperature at a time,
  * overlays the Gibbs-ensemble coexistence data of Smit & Frenkel,
    J. Chem. Phys. 94, 5663 (1991), Table II (truncated and shifted at 2.5 sigma,
    the same potential this engine samples), whose estimate is T_c = 0.459(1),
    rho_c = 0.35(1).

Usage:
  python phase3/combine_scans.py phase3/results/N16384 phase3/results/N65536
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = 1.0 / 8.0
U_STAR_ISING = 0.6107

# Smit & Frenkel (1991) Table II: T, rho_gas, err, rho_liq, err (errors are the
# subscripted uncertainties in the last digit of the published values).
SMIT_FRENKEL = np.array([
    [0.420, 0.037, 0.005, 0.72, 0.02],
    [0.431, 0.040, 0.002, 0.68, 0.02],
    [0.435, 0.052, 0.006, 0.676, 0.006],
    [0.440, 0.055, 0.007, 0.64, 0.02],
    [0.445, 0.07, 0.01, 0.66, 0.02],
    [0.455, 0.11, 0.01, 0.60, 0.03],
    [0.456, 0.11, 0.02, 0.62, 0.02],
    [0.457, 0.12, 0.03, 0.60, 0.03],
])
SF_TC, SF_TC_ERR, SF_RHOC, SF_RHOC_ERR = 0.459, 0.001, 0.35, 0.01


def fit_coexistence(T, rg, sg, rl, sl):
    """Fit A, T_c (order parameter) then rho_c, D (diameter). Returns dict."""
    dr = rl - rg
    sd = np.hypot(sg, sl)

    def order(t, A, Tc):
        return A * np.clip(Tc - t, 1e-12, None) ** BETA

    (A, Tc), cov = curve_fit(order, T, dr, p0=(1.2, T.max() + 0.02), sigma=sd, absolute_sigma=True)
    diam = 0.5 * (rl + rg)
    w = 1.0 / (0.5 * sd) ** 2
    D, rho_c = np.polyfit(T - Tc, diam, 1, w=np.sqrt(w))
    return {"Tc": Tc, "Tc_stat": float(np.sqrt(cov[1, 1])), "A": A, "rho_c": rho_c, "D": D}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="directories containing summary.json")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    scans = []
    for r in a.results:
        s = json.load(open(os.path.join(r, "summary.json")))
        s["name"] = os.path.basename(os.path.normpath(r))
        scans.append(s)

    # Pool coexistence points; floor each error at the density resolution of one particle.
    pts = []
    for s in scans:
        res = 1.0 / s["coex_Lb"] ** 2
        for t, rg, eg, rl, el in s["coexistence"]:
            pts.append((t, rg, max(eg, res), rl, max(el, res), s["N"]))
    pts = np.array(pts)
    T, rg, sg, rl, sl = pts[:, :5].T
    fit = fit_coexistence(T, rg, sg, rl, sl)

    # Systematic spread: drop one temperature at a time.
    loo = []
    for t in np.unique(T):
        k = T != t
        if len(np.unique(T[k])) >= 3:
            loo.append(fit_coexistence(T[k], rg[k], sg[k], rl[k], sl[k]))
    fit["Tc_sys"] = float(np.ptp([f["Tc"] for f in loo]) / 2) if loo else float("nan")
    fit["rho_c_sys"] = float(np.ptp([f["rho_c"] for f in loo]) / 2) if loo else float("nan")
    fit["n_points"] = int(len(T))

    out = {"fit_beta_1_8": fit, "points": pts.tolist(),
           "smit_frenkel_1991": {"Tc": SF_TC, "Tc_err": SF_TC_ERR, "rho_c": SF_RHOC, "rho_c_err": SF_RHOC_ERR}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "critical_point.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))

    sf = SMIT_FRENKEL
    ax[0].errorbar(sf[:, 1], sf[:, 0], xerr=sf[:, 2], fmt="s", mfc="none", color="gray", capsize=2,
                   label="Smit & Frenkel 1991 (Gibbs ensemble)")
    ax[0].errorbar(sf[:, 3], sf[:, 0], xerr=sf[:, 4], fmt="s", mfc="none", color="gray", capsize=2)
    ax[0].errorbar(SF_RHOC, SF_TC, xerr=SF_RHOC_ERR, yerr=SF_TC_ERR, fmt="D", color="gray", ms=8,
                   label=f"S&F critical point ({SF_TC}, {SF_RHOC})")
    markers = {16384: "o", 65536: "^"}
    for n in np.unique(pts[:, 5]):
        k = pts[:, 5] == n
        mk = markers.get(int(n), "o")
        ax[0].errorbar(rg[k], T[k], xerr=sg[k], fmt=mk, color="tab:blue", capsize=2, label=f"this work, N={int(n):,} (gas)")
        ax[0].errorbar(rl[k], T[k], xerr=sl[k], fmt=mk, color="tab:red", capsize=2, label=f"this work, N={int(n):,} (liquid)")
    tt = np.linspace(min(T.min(), sf[:, 0].min()) - 0.01, fit["Tc"], 300)
    half = 0.5 * fit["A"] * (fit["Tc"] - tt) ** BETA
    mid = fit["rho_c"] + fit["D"] * (tt - fit["Tc"])
    ax[0].plot(mid - half, tt, "k-", lw=1.2)
    ax[0].plot(mid + half, tt, "k-", lw=1.2, label=r"fit to this work, $\beta$ = 1/8")
    ax[0].plot(mid, tt, "k:", lw=1)
    tc_err = np.hypot(fit["Tc_stat"], fit["Tc_sys"])
    ax[0].plot(fit["rho_c"], fit["Tc"], "k*", ms=16,
               label=f"this work: $T_c$ = {fit['Tc']:.3f} $\\pm$ {tc_err:.3f}, $\\rho_c$ = {fit['rho_c']:.3f}")
    ax[0].set_xlabel(r"density $\rho$")
    ax[0].set_ylabel("T")
    ax[0].set_title("Liquid-gas coexistence curve (r$_c$ = 2.5, truncated & shifted)")
    ax[0].legend(fontsize=7.5, loc="lower center")
    ax[0].grid(alpha=0.4)

    # Binder cumulants from the scan with the largest box (sub-boxes smallest relative to L).
    big = max(scans, key=lambda s: s["L"])
    Tb = np.array(big["temps"])
    items = sorted(big["binder"].items(), key=lambda kv: float(kv[0].split("=")[1]))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(items)))
    for (key, v), col in zip(items, colors):
        lb = float(key.split("=")[1])
        ax[1].errorbar(Tb, v["U"], v["err"], marker="o", ms=4, capsize=2, color=col,
                       label=f"$L_b$ = {lb:.1f} ($L/L_b$ = {big['L'] / lb:.0f})")
    ax[1].axhline(U_STAR_ISING, color="k", ls=":", label="2D Ising $U^*$ (periodic box)")
    ax[1].axvline(SF_TC, color="gray", ls="--", label="S&F $T_c$")
    ax[1].set_xlabel("T")
    ax[1].set_ylabel("Binder cumulant of sub-box density")
    ax[1].set_title(f"Sub-box Binder cumulants, N={big['N']:,}, L={big['L']:.0f}")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.4)

    for s, mk in zip(scans, ["o", "^", "s"]):
        ax[2].errorbar(s["temps"], s["Cv_per_particle"], s["Cv_err"], marker=mk, capsize=3,
                       label=f"N={s['N']:,}, $\\rho$={s['rho']}")
    ax[2].axvline(SF_TC, color="gray", ls="--", label="S&F $T_c$")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$C_v^{conf}/N = N\,\mathrm{var}(E/N)/T^2$")
    ax[2].set_title("Configurational heat capacity (equilibrated runs)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.4)

    fig.tight_layout()
    path = os.path.join(a.out, "phase_diagram.png")
    fig.savefig(path, dpi=150)
    print(json.dumps(fit, indent=2, default=float))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
