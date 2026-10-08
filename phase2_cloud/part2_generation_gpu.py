"""Phase 2: large-N cooling quench of the 2D LJ fluid on the GPU.

Uses the O(N) neighbour-list engine in lj2d/ (the original all-pairs engine
needed ~1.2 s per step at N=32768 on a T4; this one needs ~1-2 ms at N=65536).

Schedule: burn-in at T_hot, then a linear ramp T_hot -> T_cold. The default
ramp crosses the liquid-gas critical region (T_c ~ 0.46 for rc = 2.5) slowly
and ends below the triple point, instead of the original 2.0 -> 0.0 ramp that
spent most of its time in the supercritical gas and finished at T = 0.

Output keys match what lj2d.dashboard expects:
  traj, T_act, T_targ, E, P, m, coord, L, N, burn_in_frames

Example:
  python phase2_cloud/part2_generation_gpu.py --N 65536 --out /tmp/part2_quench.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lj2d import LJSystem, largest_cluster_fraction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=65536)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--nu", type=float, default=0.5, help="Andersen collision frequency")
    ap.add_argument("--T-hot", type=float, default=1.0)
    ap.add_argument("--T-cold", type=float, default=0.30)
    ap.add_argument("--burn-in-steps", type=int, default=50000)
    ap.add_argument("--ramp-steps", type=int, default=2000000)
    ap.add_argument("--hold-steps", type=int, default=200000, help="extra steps at T_cold after the ramp")
    ap.add_argument("--frames", type=int, default=1200, help="approximate number of saved frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="part2_quench.npz")
    a = ap.parse_args()

    total = a.burn_in_steps + a.ramp_steps + a.hold_steps
    save_every = max(1, total // a.frames)
    sim = LJSystem(a.N, a.rho, B=1, seed=a.seed)
    print(f"N={a.N} rho={a.rho} L={sim.L:.2f} steps={total} save_every={save_every} on {sim.device}")
    sim.init_lattice(a.T_hot)

    def target(step):
        if step < a.burn_in_steps:
            return a.T_hot
        x = min((step - a.burn_in_steps) / a.ramp_steps, 1.0)
        return a.T_hot + x * (a.T_cold - a.T_hot)

    rec = {k: [] for k in ("traj", "T_act", "T_targ", "E", "P", "m", "coord")}
    t0 = time.time()
    for step in range(total + 1):
        if step % save_every == 0:
            m, coord = largest_cluster_fraction(sim)
            rec["traj"].append(sim.pos[0].cpu().numpy().astype(np.float16))
            rec["coord"].append(coord[0])
            rec["m"].append(m[0])
            rec["T_act"].append(sim.temperature().item())
            rec["T_targ"].append(target(step))
            rec["E"].append(sim.pe.item() / a.N)
            rec["P"].append(sim.pressure().item())
            if len(rec["m"]) % 50 == 1:
                rate = step / max(time.time() - t0, 1e-9)
                eta = (total - step) / max(rate, 1e-9) / 60
                print(f"step {step:8d}/{total} | T_targ {target(step):.3f} T {rec['T_act'][-1]:.3f} | "
                      f"E/N {rec['E'][-1]:.3f} | m {m[0]:.3f} | {rate:.0f} steps/s | ETA {eta:.1f} min", flush=True)
        if step < total:
            sim.step(a.dt, target(step), a.nu)

    np.savez_compressed(a.out, **{k: np.array(v) for k, v in rec.items()},
                        L=sim.L, N=a.N, burn_in_frames=a.burn_in_steps // save_every + 1,
                        config=json.dumps(vars(a)))
    print(f"Saved {a.out} ({len(rec['m'])} frames) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
