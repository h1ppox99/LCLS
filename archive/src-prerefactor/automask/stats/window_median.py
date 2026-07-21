"""
stats/window_median.py -- local tile-median-subtraction residual statistic.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from stats.base import StatSpec, register_stat, robust_z


@dataclass
class WindowMedianParams:
    win: int = 21           # tile size
    k: float = 5.0          # robust-MAD threshold
    mode: str = "low"


def window_median_stat(lit, real, win=21):
    """Local tile-median-subtraction residual as a signed robust-MAD z-score.
    Tiles the frame into win x win blocks, subtracts each block's median (over
    `real` pixels) to flatten slow illumination, then z-scores the residual.
    `lit` MUST be a genuine lit-beam frame (umean), not the dark `mean`."""
    H, W = lit.shape
    R = np.zeros((H, W), dtype=np.float64)
    for i in range(0, H, win):
        i1 = min(i + win, H)
        for j in range(0, W, win):
            j1 = min(j + win, W)
            tile, tr = lit[i:i1, j:j1], real[i:i1, j:j1]
            v = tile[tr]
            R[i:i1, j:j1] = tile - (np.median(v) if v.size else 0.0)
    return robust_z(R, real)


def compute(sample, params: WindowMedianParams | None = None):
    win = 21 if params is None else params.win
    return window_median_stat(sample.umean, sample.real, win=win)


register_stat(StatSpec(
    name="window_median",
    compute=compute,
    params=WindowMedianParams,
    kind="field",
    mode="low",
    needs=("umean", "real"),
    doc="tile-median-subtracted residual z-score; needs lit-beam umean",
))
