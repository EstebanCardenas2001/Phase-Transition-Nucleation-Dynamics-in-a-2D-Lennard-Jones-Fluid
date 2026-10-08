"""Phase 3 data generation: equilibrium NVT sampling across a temperature scan.

Every temperature is an independent replica, advanced together on the GPU.
For each sampled configuration the script stores
  * particle counts in n x n grids of sub-boxes (subsystem-block method,
    Rovere, Heermann & Binder 1990) for several n,
  * potential energy per particle and the largest-cluster fraction.

Output is written in chunks next to a checkpoint, so a run can be stopped and
resumed with the same command. Example:

  python phase3/run_critical_scan.py --N 16384 --rho 0.35 \
      --temps 0.40:0.56:9 --equil-steps 400000 --prod-steps 1600000 \
      --out /tmp/lj_scan/N16384
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lj2d import LJSystem, largest_cluster_fraction, block_density_moments  # noqa: E402


def parse_temps(spec):
    if ":" in spec:
        lo, hi, n = spec.split(":")
        return np.linspace(float(lo), float(hi), int(n))
    return np.array([float(t) for t in spec.split(",")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=16384)
    ap.add_argument("--rho", type=float, default=0.35)
    ap.add_argument("--temps", default="0.40:0.56:9", help="lo:hi:n or comma list")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--nu", type=float, default=0.1, help="Andersen collision frequency")
    ap.add_argument("--T-melt", type=float, default=1.0, help="initial melting temperature")
    ap.add_argument("--melt-steps", type=int, default=5000)
    ap.add_argument("--equil-steps", type=int, default=400000)
    ap.add_argument("--prod-steps", type=int, default=1600000)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--energy-every", type=int, default=50)
    ap.add_argument("--subdivisions", default="4,6,8,12,16,24")
    ap.add_argument("--chunk-steps", type=int, default=100000, help="steps between checkpoints")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    temps = parse_temps(args.temps)
    subdivs = [int(s) for s in args.subdivisions.split(",")]
    ckpt_path = os.path.join(args.out, "checkpoint.pt")

    sys_ = LJSystem(args.N, args.rho, B=len(temps), seed=args.seed)
    T = torch.tensor(temps, dtype=sys_.dtype, device=sys_.device)
    total = args.equil_steps + args.prod_steps

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=sys_.device)
        sys_.set_state(ck["pos"], ck["vel"])
        sys_.gen.set_state(ck["rng"].cpu() if sys_.device.type == "cpu" else ck["rng"])
        step, chunk_id = ck["step"], ck["chunk_id"]
        print(f"Resumed at step {step}/{total}")
    else:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump({**vars(args), "temps": temps.tolist(), "L": sys_.L, "rc": sys_.rc,
                       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, f, indent=2)
        sys_.init_lattice(args.T_melt)
        sys_.run(args.melt_steps, args.dt, T=args.T_melt, nu=1.0)
        step, chunk_id = 0, 0
    print(f"N={args.N} rho={args.rho} L={sys_.L:.2f} B={len(temps)} T={np.round(temps, 4).tolist()}")

    t0 = time.time()
    step0 = step
    while step < total:
        chunk_end = min(step + args.chunk_steps, total)
        rec = {"step": [], "E": [], "T_kin": [], "P": [], "s_step": [], "m": [],
               **{f"counts_{n}": [] for n in subdivs}}
        while step < chunk_end:
            sys_.step(args.dt, T, args.nu)
            step += 1
            if step % args.energy_every == 0:
                rec["step"].append(step)
                rec["E"].append((sys_.pe / args.N).cpu().numpy())
                rec["T_kin"].append(sys_.temperature().cpu().numpy())
                rec["P"].append(sys_.pressure().cpu().numpy())
            if step % args.sample_every == 0:
                pos = sys_.pos.cpu().numpy()
                rec["s_step"].append(step)
                m, _ = largest_cluster_fraction(sys_)
                rec["m"].append(m)
                for n in subdivs:
                    rec[f"counts_{n}"].append(block_density_moments(pos, sys_.L, n).astype(np.int16))
        np.savez_compressed(os.path.join(args.out, f"chunk_{chunk_id:04d}.npz"),
                            **{k: np.array(v) for k, v in rec.items()})
        chunk_id += 1
        torch.save({"pos": sys_.pos, "vel": sys_.vel, "rng": sys_.gen.get_state(),
                    "step": step, "chunk_id": chunk_id}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

        rate = (step - step0) / (time.time() - t0)
        eta = (total - step) / rate / 3600
        phase = "EQUIL" if step <= args.equil_steps else "PROD"
        e_now = np.round(rec["E"][-1], 3).tolist()
        print(f"[{phase}] step {step}/{total} | {rate:.0f} steps/s | ETA {eta:.2f} h | E/N {e_now}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
