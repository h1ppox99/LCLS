"""
stats/variance.py -- low-variance statistic (dead/shadowed/beam-stop pixels).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat, robust_z


@dataclass
class VarianceParams:
    k: float = 3.5          # robust-MAD threshold
    mode: str = "low"       # defects show as LOW-variance islands on a lit run


def variance_stat(std):
    """Signed robust-MAD z-score of log10(per-pixel standard deviation).

    z << 0 = low-variance (dead/shadowed/beam-stop), z >> 0 = high-variance. On a
    lit run photon shot noise lifts live pixels, so defects show as LOW-variance
    islands -- thresholded mode="low"."""
    return robust_z(std, std > 0, transform=np.log10)


def compute(sample, params: VarianceParams | None = None):
    return variance_stat(sample.std)


register_stat(StatSpec(
    name="variance",
    compute=compute,
    params=VarianceParams,
    kind="field",
    mode="low",
    needs=("std",),
    doc="robust-MAD z of log10 per-pixel std; low z == dead/shadowed",
))
