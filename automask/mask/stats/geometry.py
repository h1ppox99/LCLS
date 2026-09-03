"""
stats/geometry.py -- geometry floor statistic (ASIC boundaries + module gap).

Intensity-free, 100%-precision. This is a FLOOR stat: it emits a boolean mask
directly and carries no tunable parameters.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.mask.stats.base import StatSpec, register_stat
from automask.mask.regularization.pad import pad_mask


@dataclass
class GeometryParams:
    pad: int = 2
    frac: float = 0.4


def geometry_mask(real: np.ndarray, pad: int = 2, frac: float = 0.4) -> np.ndarray:
    """Mask assembled border lines: rows/cols where more than `frac` of pixels
    are UNMAPPED (ASIC boundaries + inter-module gap), widened by `pad` px each
    side. `real` is the boolean map of pixels carrying a real value (sum != 0)."""
    H, W = real.shape
    mask = np.zeros((H, W), bool)
    mask[(~real).mean(1) > frac, :] = True
    mask[:, (~real).mean(0) > frac] = True
    return pad_mask(mask, pad)


def compute(sample, params: GeometryParams | None = None):
    p = params or GeometryParams()
    return geometry_mask(sample.real, pad=p.pad, frac=p.frac)


register_stat(
    StatSpec(
        name="geometry",
        compute=compute,
        params=GeometryParams,
        kind="floor",
        needs=("real",),
        doc="ASIC/gap border lines from the unmapped-pixel map; 100%-precision floor",
    )
)
