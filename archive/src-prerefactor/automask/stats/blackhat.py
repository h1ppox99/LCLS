"""
stats/blackhat.py -- black-hat (dark-speck) statistic.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from skimage.morphology import black_tophat, disk

from stats.base import StatSpec, register_stat, robust_z


@dataclass
class BlackhatParams:
    radius: int = 5         # structuring-element radius
    k: float = 6.0          # robust-MAD threshold
    mode: str = "high"      # black-hat responds one-sided (R >= 0)


def blackhat_stat(lit, real, radius=5):
    """Black-hat response as a robust-MAD z-score: closing(lit) - lit over `real`.
    Responds to pixels DARKER than their local surroundings (dead spots, dark
    specks, beam-stop edge). One-sided (R >= 0), so threshold mode="high"."""
    filled = np.where(real, lit, np.median(lit[real]))
    R = black_tophat(filled, footprint=disk(radius))
    return robust_z(R, real)


def compute(sample, params: BlackhatParams | None = None):
    radius = 5 if params is None else params.radius
    return blackhat_stat(sample.umean, sample.real, radius=radius)


register_stat(StatSpec(
    name="blackhat",
    compute=compute,
    params=BlackhatParams,
    kind="field",
    mode="high",
    needs=("umean", "real"),
    doc="grey black-hat response z-score; needs lit-beam umean",
))
