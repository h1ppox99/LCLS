"""Calibration-status bad-pixel floor statistic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat
from automask.regularization.pad import pad_mask


@dataclass
class CalibParams:
    """`pad` widens each flagged pixel by `pad` px per side with a (2*pad+1)
    square dilation, matching the lab baseline's 5x5 dead-pixel dilation (pad=2)
    and the geometry floor stat. A dead pixel corrupts its neighbours (charge
    sharing / interpolation), so the halo is genuinely bad, not padding slack."""
    pad: int = 2


def compute(sample, params: CalibParams | None = None):
    """Return the frozen psana pixel-status mask (``True == masked``), dilated by
    ``pad`` px per side."""
    p = params or CalibParams()
    return pad_mask(np.asarray(sample.calib, dtype=bool), p.pad)


register_stat(StatSpec(
    name="calib",
    compute=compute,
    params=CalibParams,
    kind="floor",
    needs=("calib",),
    doc="psana pixel-status bad-pixel mask, dilated; intensity-free floor",
))
