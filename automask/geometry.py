#!/usr/bin/env python3
"""
geometry.py -- beam-center lookup for the frozen assembled arrays.

Reads a run's small-data HDF5 directly (like producers/*.py do -- this is the
one place outside producers/ that needs h5py) to get the psana-fitted beam
center (`azav__azav_xcen/ycen`, mm) and the `ix`/`iy` panel->assembled pixel
maps, then reports the center as an exact pixel index rather than a
hand-rounded approximation.

Axis-order gotcha: CLAUDE.md's "(col 1005, row 45)" describes the true
assembled image (1030 rows x 1064 cols). But every `_asm` array frozen by
producers/extract_dataset.py and features/store.py is built with
`out[ix, iy] = panel` on an array of shape (1064, 1030) -- i.e. TRANSPOSED:
axis0 is the ix/"column" index (0..1063), axis1 is the iy/"row" index
(0..1029). Verified directly: sumimg[1005, 45] equals the raw panel value at
the pixel nearest (xcen, ycen); sumimg[45, 1005] does not. So for THESE
arrays the center is (1005, 45), not (45, 1005) -- get_center() returns it in
this axis0/axis1 order so callers can plug it straight into
`np.indices(sumimg.shape)` without re-deriving the swap.
"""
from __future__ import annotations
import os
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../LCLS
SMALLDATA = os.path.join(ROOT, "hdf5", "smalldata")
DET = "jungfrau1M_alcove"


def get_center(run: int, det: str = DET) -> tuple[float, float]:
    """Beam center in assembled-array (axis0, axis1) index order -- matches
    `dataset.load_image()`'s `(1064, 1030)` shape directly.

    Finds the native (panel) pixel closest to the psana-fitted (xcen, ycen)
    [mm] via the per-pixel lab-frame x/y maps (both um), then reads off that
    pixel's (ix, iy) assembled indices.
    """
    path = os.path.join(SMALLDATA, f"xppl1016922_Run{run:04d}.h5")
    with h5py.File(path, "r") as f:
        g = f[f"UserDataCfg/{det}"]
        xcen_mm = float(g["azav__azav_xcen"][()][0])
        ycen_mm = float(g["azav__azav_ycen"][()][0])
        ix = g["ix"][()].astype(np.int64)
        iy = g["iy"][()].astype(np.int64)
        x = g["x"][()]   # um, native (2,512,1024)
        y = g["y"][()]   # um, native (2,512,1024)

    d2 = (x - xcen_mm * 1000.0) ** 2 + (y - ycen_mm * 1000.0) ** 2
    idx = np.unravel_index(np.argmin(d2), d2.shape)
    return float(ix[idx]), float(iy[idx])


if __name__ == "__main__":
    import sys
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 475
    c0, c1 = get_center(run)
    print(f"run {run}: assembled center (axis0, axis1) = ({c0:.1f}, {c1:.1f})")
