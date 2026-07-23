"""
stats/dead_holes.py -- isolated dead-pixel / interior-hole floor statistic.

Intensity-free, high-precision floor. Complements `geometry` (which masks the
large ASIC/module gaps as whole rows/cols): this flags the OPPOSITE case --
individual not-real pixels ("holes") that sit inside the live area and would
otherwise be masked by nothing unless psana's `calib` status happens to know
them. A not-real pixel whose small neighborhood is mostly live is, physically, a
dead/bad pixel inside the illuminated region.

See exploration.MD finding F1: without this, an isolated newly-dead pixel absent
from `calib` is caught by no stat (the intensity detectors are all `real`-gated).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from automask.stats.base import StatSpec, register_stat


@dataclass
class DeadHolesParams:
    win: int = 5             # neighborhood size for the local live-fraction
    min_live_frac: float = 0.6   # a hole must sit in a >= this-live neighborhood


def dead_holes_mask(real: np.ndarray, win: int = 5, min_live_frac: float = 0.6):
    """Not-real pixels enclosed by a mostly-live neighborhood (isolated holes).

    The local live fraction is computed with a `win`x`win` box filter; large gaps
    (borders, module gap) have a low live fraction and are left to `geometry`,
    while isolated holes sit in a live surround and are flagged here."""
    live = uniform_filter(real.astype(np.float64), size=win)
    return (~real) & (live >= min_live_frac)


def compute(sample, params: DeadHolesParams | None = None):
    p = params or DeadHolesParams()
    return dead_holes_mask(sample.real, win=p.win, min_live_frac=p.min_live_frac)


register_stat(StatSpec(
    name="dead_holes",
    compute=compute,
    params=DeadHolesParams,
    kind="floor",
    needs=("real",),
    doc="isolated not-real pixels enclosed by live area; intensity-free floor",
))
