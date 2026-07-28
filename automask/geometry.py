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
HERE = os.path.dirname(os.path.abspath(__file__))
GEOM_DIR = os.path.join(HERE, "data", "geometry")
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


def index_maps(run: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen `(ix, iy)` panel->assembled index maps, shape (2, 512, 1024) each.

    Pure numpy: reads the arrays written by
    `python -m automask.producers.freeze_geometry`, so stats that need native
    panel geometry don't pull h5py/psana into the evaluation loop.
    """
    px = os.path.join(GEOM_DIR, f"ix_run{run:04d}.npy")
    py = os.path.join(GEOM_DIR, f"iy_run{run:04d}.npy")
    if not (os.path.exists(px) and os.path.exists(py)):
        raise FileNotFoundError(
            f"no frozen index maps for run {run} in {GEOM_DIR}; run "
            f"`python -m automask.producers.freeze_geometry {run}` first")
    return np.load(px), np.load(py)


def asm_shape(run: int) -> tuple[int, int]:
    """Assembled-canvas shape for `run`, derived from its frozen index maps
    rather than hardcoded: `(ix.max()+1, iy.max()+1)` is exactly the canvas
    `out[ix, iy] = panel` needs to hold every pixel the maps place. Read from
    data for the same reason `get_center` is -- if a future geometry epoch
    changes the canvas size, this stays correct without a code edit."""
    ix, iy = index_maps(run)
    return int(ix.max()) + 1, int(iy.max()) + 1


def panel_to_asm(panel: np.ndarray, run: int, fill=0) -> np.ndarray:
    """Scatter a native `(2, 512, 1024)` array into this run's assembled canvas.

    Mirrors `out[ix, iy] = panel`, the construction every frozen `_asm` array
    uses (see the axis-order note above). Pixels no panel maps onto keep `fill`.
    """
    ix, iy = index_maps(run)
    if panel.shape != ix.shape:
        raise ValueError(f"expected panel shape {ix.shape}, got {panel.shape}")
    shape = (int(ix.max()) + 1, int(iy.max()) + 1)
    out = np.full(shape, fill, dtype=panel.dtype)
    out[ix, iy] = panel
    return out


def asm_to_panel(asm: np.ndarray, run: int) -> np.ndarray:
    """Gather an assembled-canvas array back into native `(2, 512, 1024)`.

    The inverse of `panel_to_asm` on the pixels the panels cover; assembled
    pixels outside any panel are simply not read.
    """
    ix, iy = index_maps(run)
    shape = (int(ix.max()) + 1, int(iy.max()) + 1)
    if asm.shape != shape:
        raise ValueError(f"expected assembled shape {shape}, got {asm.shape}")
    return asm[ix, iy]


if __name__ == "__main__":
    import sys
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 475
    c0, c1 = get_center(run)
    print(f"run {run}: assembled center (axis0, axis1) = ({c0:.1f}, {c1:.1f})")
