#!/usr/bin/env python3
"""
geometry.py -- Jungfrau panel/assembled geometry through psana.

Runtime uses the experiment's fixed beam center and psana's official panel
index maps.

Axis-order gotcha: CLAUDE.md's "(col 1005, row 45)" describes the true
assembled image (1030 rows x 1064 cols). Every project `_asm` array is built with
`out[ix, iy] = panel` on an array of shape (1064, 1030) -- i.e. TRANSPOSED:
axis0 is the ix/"column" index (0..1063), axis1 is the iy/"row" index
(0..1029). Verified directly: image[1005, 45] equals the raw panel value at
the pixel nearest (xcen, ycen); image[45, 1005] does not. So for THESE
arrays the center is (1005, 45), not (45, 1005) -- get_center() returns it in
this axis0/axis1 order so callers can plug it straight into
`np.indices(image.shape)` without re-deriving the swap.
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np

DET = "jungfrau1M_alcove"
BEAM_CENTER = (1005.0, 45.0)


def get_center(run: int, det: str = DET) -> tuple[float, float]:
    """Beam center in assembled-array (axis0, axis1) index order -- matches
    the `(1064, 1030)` assembled arrays ImageStore serves.

    This experiment used one fixed Jungfrau geometry and beam center.
    """
    return BEAM_CENTER


@lru_cache(maxsize=None)
def index_maps(run: int) -> tuple[np.ndarray, np.ndarray]:
    """Official psana `(ix, iy)` maps, shape `(2, 512, 1024)` each."""
    from automask.io.read_xtc import panel_geometry

    return panel_geometry(run)


def asm_shape(run: int) -> tuple[int, int]:
    """Assembled-canvas shape for `run`, derived from its psana index maps
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
