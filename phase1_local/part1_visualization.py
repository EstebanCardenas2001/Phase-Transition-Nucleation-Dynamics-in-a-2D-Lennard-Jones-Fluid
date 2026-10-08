"""Render the Phase 1 dashboard video from part1_ladder_massive.npz.

Thin wrapper around lj2d.dashboard; run from anywhere:
  python phase1_local/part1_visualization.py [--out massive_nucleation_dashboard.mp4]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lj2d.dashboard import render  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "part1_ladder_massive.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "massive_nucleation_dashboard.mp4"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    render(a.npz, a.out, zoom_frac=1 / 3, skip=2, dpi=200, workers=a.workers)
