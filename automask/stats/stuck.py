"""
stats/stuck.py -- stuck-pixel (zero temporal variance) floor statistic.

Intensity-free of the mean, high-precision floor. A live pixel under X-ray
illumination must show shot-to-shot photon noise, so a data-carrying pixel with
essentially zero per-pixel std (`ustd ~ 0`) is stuck (hot-stuck or frozen at a
constant value) regardless of how bright it reads. This is the polarity the
`variance` field stat cannot see: `variance` scores `robust_z(ustd, ustd>0)` and
so EXCLUDES exactly-zero-variance pixels (exploration.MD F1).

Kept separate from `variance` (which stays a graded low-variance detector) so the
production recipe is unchanged; this adds only the degenerate `ustd == 0` case.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat


@dataclass
class StuckParams:
    eps: float = 0.0         # ustd <= eps counts as stuck (0.0 == exactly frozen)


def stuck_mask(real: np.ndarray, ustd: np.ndarray, eps: float = 0.0):
    """Real (data-carrying) pixels whose per-shot std is <= eps == stuck."""
    return real & (ustd <= eps)


def compute(sample, params: StuckParams | None = None):
    p = params or StuckParams()
    return stuck_mask(sample.real, sample.ustd, eps=p.eps)


register_stat(StatSpec(
    name="stuck",
    compute=compute,
    params=StuckParams,
    kind="floor",
    needs=("real", "ustd"),
    doc="data-carrying pixels with ~zero temporal variance (stuck); floor",
))
