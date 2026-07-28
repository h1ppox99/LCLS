#!/usr/bin/env python3
"""
producers/freeze_geometry.py -- freeze the panel->assembled index maps to .npy.

The `ix`/`iy` maps live in each run's small-data `UserDataCfg/<det>/` (the same
place `geometry.get_center` reads them from), so freezing them needs h5py but not
psana. Once frozen, `automask.geometry.panel_to_asm` / `asm_to_panel` are pure
numpy -- which is what lets a stat work in native panel geometry (per-ASIC
structure) while still emitting an assembled-space field, without dragging psana
into the evaluation loop.

Run:  python -m automask.producers.freeze_geometry
"""
from __future__ import annotations
import os

import h5py
import numpy as np

from automask.evaluation import EVAL_RUNS
from automask.geometry import DET, GEOM_DIR, SMALLDATA


def freeze(run: int, det: str = DET) -> tuple[str, str]:
    path = os.path.join(SMALLDATA, f"xppl1016922_Run{run:04d}.h5")
    with h5py.File(path, "r") as f:
        g = f[f"UserDataCfg/{det}"]
        ix = g["ix"][()].astype(np.int64)
        iy = g["iy"][()].astype(np.int64)
    os.makedirs(GEOM_DIR, exist_ok=True)
    px = os.path.join(GEOM_DIR, f"ix_run{run:04d}.npy")
    py = os.path.join(GEOM_DIR, f"iy_run{run:04d}.npy")
    np.save(px, ix)
    np.save(py, iy)
    print(f"run {run}: ix/iy {ix.shape} -> {px}")
    return px, py


def main(runs=None):
    for run in (EVAL_RUNS if runs is None else runs):
        freeze(run)


if __name__ == "__main__":
    import sys
    main([int(a) for a in sys.argv[1:]] or None)
