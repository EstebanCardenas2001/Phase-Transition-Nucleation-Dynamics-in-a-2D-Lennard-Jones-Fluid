"""Render the Phase 2 dashboard video from the quench trajectory.

Thin wrapper around lj2d.dashboard:
  python phase2_cloud/part2_visualization.py --npz /tmp/lj_phase2/part2_quench_N65536.npz
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lj2d.dashboard import render  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "part2_quench.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "quench_dashboard.mp4"))
    ap.add_argument("--zoom-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    render(a.npz, a.out, zoom_frac=a.zoom_frac, dpi=200, workers=a.workers)
