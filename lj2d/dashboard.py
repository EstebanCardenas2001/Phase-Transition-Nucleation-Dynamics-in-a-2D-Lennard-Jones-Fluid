"""Five-panel diagnostic dashboard rendered from a trajectory .npz file.

Expected keys: traj [F, N, 2], T_act, T_targ, E, m [F], coord [F, N], L, N, burn_in_frames.

Panels: global box, magnified view of the box centre (optionally tracking the densest region), coordination
histogram, energy + temperature traces, and the largest-cluster order parameter.
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


def dense_region_track(traj, coord, L, n_grid=64, min_coord=4, smooth=0.15):
    """Per-frame centre of the densest region, smoothed in time on the periodic torus.

    Particles with coordination >= min_coord are histogrammed on a periodic grid,
    blurred, and the maximum is taken. The centre then moves by an exponential
    moving average (factor `smooth`) along the shortest periodic path so the
    camera glides rather than jumps between droplets.
    """
    centres = np.empty((len(traj), 2))
    c = np.array([L / 2, L / 2])
    edges = np.linspace(0, L, n_grid + 1)
    for f, (pos, k) in enumerate(zip(traj, coord)):
        dense = pos[k >= min_coord]
        if len(dense) > 0:
            h, _, _ = np.histogram2d(dense[:, 0], dense[:, 1], bins=[edges, edges])
            h = gaussian_filter(h, sigma=2.0, mode="wrap")
            ix, iy = np.unravel_index(np.argmax(h), h.shape)
            target = (np.array([ix, iy]) + 0.5) * (L / n_grid)
            step = target - c
            step -= L * np.round(step / L)
            c = (c + smooth * step) % L
        centres[f] = c
    return centres


def render(npz_path, out_path, zoom_frac=1 / 4, skip=1, fps=30, dpi=200, figsize=(18, 12),
           marker_size=None, max_frames=None, track=False, workers=1):
    """Render the dashboard. With workers > 1 (mp4 only) frame ranges are drawn in
    parallel processes and joined losslessly with ffmpeg's concat demuxer."""
    kw = dict(zoom_frac=zoom_frac, skip=skip, fps=fps, dpi=dpi, figsize=figsize,
              marker_size=marker_size, max_frames=max_frames, track=track)
    if workers <= 1 or out_path.endswith(".gif"):
        return _render_range(npz_path, out_path, None, **kw)
    n_frames = len(np.load(npz_path)["T_act"][::skip][:max_frames])
    bounds = np.linspace(0, n_frames, workers + 1).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        parts = [os.path.join(tmp, f"part{i:03d}.mp4") for i in range(workers)]
        with ProcessPoolExecutor(workers) as ex:
            futures = [ex.submit(_render_range, npz_path, part, (a, b), **kw)
                       for part, a, b in zip(parts, bounds[:-1], bounds[1:])]
            for fut in futures:
                fut.result()
        listing = os.path.join(tmp, "parts.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{part}'\n" for part in parts)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", listing, "-c", "copy", out_path], check=True)
    print(f"Saved {out_path}")


def _render_range(npz_path, out_path, frame_range, zoom_frac, skip, fps, dpi, figsize,
                  marker_size, max_frames, track):
    data = np.load(npz_path)
    traj = data["traj"][::skip].astype(np.float32)
    coord = data["coord"][::skip]
    T_act, T_targ = data["T_act"][::skip], data["T_targ"][::skip]
    E, m = data["E"][::skip], data["m"][::skip]
    L, N = float(data["L"]), int(data["N"])
    burn = int(data["burn_in_frames"]) // skip
    if max_frames:
        traj, coord, T_act, T_targ, E, m = (a[:max_frames] for a in (traj, coord, T_act, T_targ, E, m))
    n_frames = len(traj)
    n_prod = n_frames - burn
    t_axis = np.arange(n_prod)
    frames = range(*frame_range) if frame_range else range(n_frames)
    print(f"Rendering frames {frames.start}-{frames.stop - 1} of {n_frames} ({burn} burn-in), N={N} -> {out_path}")

    zoom_w = L * zoom_frac
    centres = dense_region_track(traj, coord, L) if track else np.full((n_frames, 2), L / 2)

    prod = slice(burn, None)
    e_lo, e_hi = E[prod].min(), E[prod].max()
    e_pad = 0.1 * (e_hi - e_lo + 1e-9)
    t_hi = max(T_act[prod].max(), T_targ[prod].max())
    # Condensed fraction: share of particles with >= 4 bonded neighbours. Unlike m,
    # it grows even when the liquid is split over many separate domains.
    cond = (coord >= 4).mean(1)
    m_hi = min(1.0, max(m[prod].max(), cond[prod].max()) * 1.15 + 0.02)
    s_global = marker_size or float(np.clip(2.4e4 / N, 0.3, 8.0))
    s_zoom = s_global / zoom_frac ** 2 * 0.35

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 6, height_ratios=[1.5, 1], wspace=0.8, hspace=0.3)
    fig.subplots_adjust(right=0.92, top=0.90, bottom=0.08, left=0.06)
    cmap = matplotlib.colormaps["coolwarm"]

    def box_axis(ax, title):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
        ax.set_title(title, fontsize=14, pad=12)

    ax_sim = fig.add_subplot(gs[0, 0:3])
    ax_sim.set_xlim(0, L)
    ax_sim.set_ylim(0, L)
    box_axis(ax_sim, f"Global domain (N={N:,})")
    ax_zoom = fig.add_subplot(gs[0, 3:6])
    ax_zoom.set_xlim(-zoom_w / 2, zoom_w / 2)
    ax_zoom.set_ylim(-zoom_w / 2, zoom_w / 2)
    box_axis(ax_zoom, "Magnified view (tracking densest region)" if track else "Magnified core view")
    zoom_rect = Rectangle((0, 0), zoom_w, zoom_w, lw=1.5, ec="white", fc="none", ls="--", alpha=0.8)
    ax_sim.add_patch(zoom_rect)

    status = fig.text(0.5, 0.95, "", ha="center", va="center", fontsize=16, fontweight="bold",
                      bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", pad=8))

    ax_hist = fig.add_subplot(gs[1, 0:2])
    ax_hist.set_xlim(-0.5, 8.5)
    ax_hist.set_ylim(0, 1.05 * max(np.bincount(k.astype(np.int64), minlength=10)[:10].max() for k in coord[burn:]))
    ax_hist.set_xlabel("Local coordination number")
    ax_hist.set_ylabel("Particle count")
    ax_hist.set_title("Phase coexistence", fontsize=12)
    ax_hist.grid(True, axis="y", ls="--", alpha=0.6)
    bins = np.arange(-0.5, 10.5, 1.0)
    bars = ax_hist.bar(np.arange(10), np.zeros(10), width=0.8, edgecolor="black", alpha=0.9,
                       color=[cmap(min(i / 6.0, 1.0)) for i in range(10)])

    ax_e = fig.add_subplot(gs[1, 2:4])
    ax_e.set_xlim(0, max(n_prod, 1))
    ax_e.set_ylim(e_lo - e_pad, e_hi + e_pad)
    ax_e.set_ylabel("Potential energy / N", color="blue")
    ax_e.set_xlabel("Production frame")
    ax_e.grid(True, ls="--", alpha=0.6)
    ax_t = ax_e.twinx()
    ax_t.set_ylim(0, t_hi * 1.1)
    ax_t.set_ylabel("Temperature", color="red", labelpad=12)

    ax_m = fig.add_subplot(gs[1, 4:6])
    ax_m.set_xlim(0, max(n_prod, 1))
    ax_m.set_ylim(0, m_hi)
    ax_m.set_xlabel("Production frame")
    ax_m.set_ylabel("Fraction of particles")
    ax_m.yaxis.set_label_position("right")
    ax_m.yaxis.tick_right()
    ax_m.set_title("Structural order parameters", fontsize=12)
    ax_m.grid(True, ls="--", alpha=0.6)

    sc = ax_sim.scatter(traj[0, :, 0], traj[0, :, 1], s=s_global, c=coord[0], cmap=cmap,
                        vmin=0, vmax=6, edgecolors="none")
    sc_zoom = ax_zoom.scatter([], [], s=s_zoom, c=[], cmap=cmap, vmin=0, vmax=6, edgecolors="none")
    l_e, = ax_e.plot([], [], color="blue", lw=1.5)
    l_tt, = ax_t.plot([], [], color="black", ls="--", label="Target T")
    l_ta, = ax_t.plot([], [], color="red", alpha=0.7, label="Kinetic T")
    ax_t.legend(loc="upper right", fontsize=9)
    l_m, = ax_m.plot([], [], color="purple", lw=1.5, label="largest cluster m")
    l_c, = ax_m.plot([], [], color="darkorange", lw=1.5, label="condensed (coord $\\geq$ 4)")
    ax_m.legend(loc="upper left", fontsize=9)
    vlines = [ax_e.axvline(0, color="gray", ls=":"), ax_m.axvline(0, color="gray", ls=":")]

    def update(f):
        pos, k, c = traj[f], coord[f], centres[f]
        sc.set_offsets(pos)
        sc.set_array(k)
        # Zoom: shift so the window centre sits at the origin, wrapping periodically.
        rel = pos - c
        rel -= L * np.round(rel / L)
        inside = (np.abs(rel) < zoom_w / 2 + 1.0).all(1)
        sc_zoom.set_offsets(rel[inside])
        sc_zoom.set_array(k[inside])
        zoom_rect.set_xy(c - zoom_w / 2)

        if f < burn:
            status.set_text("BURN-IN (thermalizing)")
            status.set_color("tomato")
            for bar in bars:
                bar.set_height(0)
            for line in (l_e, l_tt, l_ta, l_m, l_c):
                line.set_data([], [])
            for v in vlines:
                v.set_visible(False)
        else:
            status.set_text(f"PRODUCTION & COOLING   T_target = {T_targ[f]:.3f}")
            status.set_color("green")
            counts, _ = np.histogram(k, bins=bins)
            for bar, n in zip(bars, counts):
                bar.set_height(n)
            p = f - burn
            x = t_axis[: p + 1]
            l_e.set_data(x, E[burn: f + 1])
            l_tt.set_data(x, T_targ[burn: f + 1])
            l_ta.set_data(x, T_act[burn: f + 1])
            l_m.set_data(x, m[burn: f + 1])
            l_c.set_data(x, cond[burn: f + 1])
            for v in vlines:
                v.set_visible(True)
                v.set_xdata([p, p])
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    if out_path.endswith(".gif"):
        ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi,
                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("out", help=".mp4 (ffmpeg) or .gif")
    ap.add_argument("--zoom-frac", type=float, default=0.25, help="zoom window side / box side")
    ap.add_argument("--skip", type=int, default=1, help="render every k-th frame")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--width", type=float, default=18, help="figure width in inches (height = 2/3)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--track", action="store_true", help="move the zoom window to follow the densest region (default: fixed at the box centre)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel render processes (mp4 only)")
    a = ap.parse_args()
    render(a.npz, a.out, zoom_frac=a.zoom_frac, skip=a.skip, fps=a.fps, dpi=a.dpi,
           figsize=(a.width, a.width * 2 / 3), max_frames=a.max_frames, track=a.track,
           workers=a.workers)


if __name__ == "__main__":
    main()
