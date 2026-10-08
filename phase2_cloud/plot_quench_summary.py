"""Static summary figures for the Phase 2 quench.

  outputs/quench_snapshots.png      the box and its centre window at five temperatures
  outputs/quench_thermodynamics.png E/N, condensed and hexagonal fractions, and dE/dT
                                    against the ramp temperature

Usage:
  python phase2_cloud/plot_quench_summary.py --npz /tmp/lj_phase2/part2_quench_N65536.npz
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

# Single-hue sequential ramp for coordination number (light = few neighbours).
COORD_CMAP = LinearSegmentedColormap.from_list(
    "coord", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BLUE, ORANGE = "#2a78d6", "#eb6834"     # categorical slots 1 and 2
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
T_C = 0.453          # this work, phase3/results/critical_point.json
T_FREEZE = 0.365     # steepest energy drop in this run


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def snapshots(d, burn, temps, zoom_frac):
    L, N = float(d["L"]), int(d["N"])
    T = d["T_targ"]
    frames = [burn + int(np.argmin(np.abs(T[burn:] - t))) for t in temps]
    frames[-1] = len(T) - 1   # last frame: end of the hold at T_cold
    w = L * zoom_frac
    fig, ax = plt.subplots(2, len(frames), figsize=(3.6 * len(frames), 7.6))
    for j, f in enumerate(frames):
        pos, k = d["traj"][f].astype(np.float32), d["coord"][f]
        order = np.argsort(k)  # dense particles drawn last
        a = ax[0, j]
        a.scatter(pos[order, 0], pos[order, 1], c=k[order], s=0.12, cmap=COORD_CMAP, vmin=0, vmax=6,
                  linewidths=0, rasterized=True)
        a.add_patch(Rectangle((L / 2 - w / 2, L / 2 - w / 2), w, w, fill=False, ec=ORANGE, lw=1.2))
        a.set_xlim(0, L)
        a.set_ylim(0, L)
        cond = (k >= 4).mean()
        a.set_title(f"T = {T[f]:.2f}\ncondensed {cond:.0%}", fontsize=11, color=INK)
        z = ax[1, j]
        rel = pos - L / 2
        inside = (np.abs(rel) < w / 2 + 1).all(1)
        o = np.argsort(k[inside])
        z.scatter(rel[inside][o, 0], rel[inside][o, 1], c=k[inside][o], s=1.6, cmap=COORD_CMAP, vmin=0, vmax=6,
                  linewidths=0, rasterized=True)
        z.set_xlim(-w / 2, w / 2)
        z.set_ylim(-w / 2, w / 2)
        for b in (a, z):
            b.set_aspect("equal")
            b.set_xticks([])
            b.set_yticks([])
            for s in b.spines.values():
                s.set_color(GRID)
    ax[0, 0].set_ylabel(f"full box (L = {L:.0f}σ)", color=MUTED)
    ax[1, 0].set_ylabel(f"centre window ({w:.0f}σ)", color=MUTED)
    sm = plt.cm.ScalarMappable(cmap=COORD_CMAP, norm=plt.Normalize(0, 6))
    cb = fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01)
    cb.set_label("coordination number (neighbours within 1.5σ)", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle(f"Phase 2 quench, N = {N:,}, ρ = 0.3: gas → liquid domains → hexagonal crystallites",
                 fontsize=13, color=INK)
    path = os.path.join(OUT, "quench_snapshots.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def thermodynamics(d, burn, cfg):
    T, E = d["T_targ"], d["E"]
    ramp = np.arange(len(T)) >= burn
    ramp &= T > cfg["T_cold"] + 1e-6           # exclude burn-in and the final hold
    T, E = T[ramp], E[ramp]
    coord = d["coord"][ramp]
    cond, hexa = (coord >= 4).mean(1), (coord == 6).mean(1)
    dEdT = uniform_filter1d(np.gradient(uniform_filter1d(E, 15), T), 25)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    ax[0].plot(T, E, color=BLUE, lw=2)
    ax[0].set_ylabel("potential energy per particle, E/N", color=INK)
    ax[1].plot(T, cond, color=BLUE, lw=2)
    ax[1].plot(T, hexa, color=ORANGE, lw=2)
    # Direct labels on the flat, uncluttered hot end of the curves.
    ax[1].text(0.97, cond[np.argmin(np.abs(T - 0.97))] + 0.05, "condensed (≥ 4 neighbours)", color=INK, fontsize=10)
    ax[1].text(0.97, hexa[np.argmin(np.abs(T - 0.97))] + 0.03, "hexagonal (exactly 6)", color=INK, fontsize=10)
    ax[1].set_ylabel("fraction of particles", color=INK)
    ax[1].set_ylim(0, 1)
    ax[2].plot(T, dEdT, color=BLUE, lw=2)
    ax[2].set_ylabel("dE/dT along the ramp", color=INK)
    titles = ["Energy", "Structure", "Energy release rate"]
    for a, title in zip(ax, titles):
        style(a)
        a.set_title(title, color=INK, fontsize=12)
        a.set_xlabel("ramp temperature T (cooling →)", color=INK)
        for t, lab, ls in ((T_C, "T$_c$", "--"), (T_FREEZE, "freezing", ":")):
            a.axvline(t, color=MUTED, ls=ls, lw=1.2)
            a.text(t, 1.0, f" {lab}", transform=a.get_xaxis_transform(), color=MUTED, fontsize=9, va="top")
    ax[0].invert_xaxis()   # shared x: cooling reads left to right
    fig.suptitle("Phase 2: condensation sets in below T$_c$; the largest energy release is freezing",
                 fontsize=13, color=INK)
    fig.tight_layout()
    path = os.path.join(OUT, "quench_thermodynamics.png")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=os.path.join(HERE, "part2_quench.npz"))
    ap.add_argument("--temps", default="0.80,0.50,0.44,0.38,0.30")
    ap.add_argument("--zoom-frac", type=float, default=0.15, help="same window as the video")
    a = ap.parse_args()
    d = np.load(a.npz)
    cfg = json.loads(str(d["config"]))
    burn = int(d["burn_in_frames"])
    temps = [float(t) for t in a.temps.split(",")]
    print("Saved", snapshots(d, burn, temps, a.zoom_frac))
    print("Saved", thermodynamics(d, burn, cfg))


if __name__ == "__main__":
    main()
