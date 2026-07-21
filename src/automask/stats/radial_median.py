"""
stats/radial_median.py -- radial-bin median-subtraction statistic (promoted).

Bins the assembled sum image into concentric annuli around the run's beam center
(sample.center; see geometry.get_center for the axis-order subtlety) and subtracts
each annulus's median from every pixel in it, flattening the radially-symmetric
scattering falloff. Local outliers that don't follow the ring pattern (hot/dead
pixels, panel artifacts) survive as large |residual|; a robust-MAD z-score of that
residual is the evidence field. Two-sided (mode="both").

Promoted from studies/radial_median_subtract.py.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from stats.base import StatSpec, register_stat, robust_z


@dataclass
class RadialMedianParams:
    bin_width: float = 3.0      # px, radial annulus width
    k: float = 5.0
    mode: str = "both"


def radial_median_stat(sumimg, real, center, bin_width=3.0):
    """Robust-MAD z-score of the radial-bin median-subtracted residual. `center`
    is (axis0, axis1) in the frozen array's own index order (sample.center)."""
    sumimg = sumimg.astype(np.float64, copy=False)
    c0, c1 = center
    i0, i1 = np.indices(sumimg.shape)
    bin_idx = (np.hypot(i0 - c0, i1 - c1) // bin_width).astype(np.int64)
    n_bins = int(bin_idx.max()) + 1
    medians = np.zeros(n_bins)
    for b in range(n_bins):
        vals = sumimg[(bin_idx == b) & real]
        medians[b] = np.median(vals) if vals.size else 0.0
    residual = sumimg - medians[bin_idx]
    residual[~real] = 0.0
    return robust_z(residual, real)


def compute(sample, params: RadialMedianParams | None = None):
    p = params or RadialMedianParams()
    return radial_median_stat(sample.sumimg, sample.real, sample.center, p.bin_width)


register_stat(StatSpec(
    name="radial_median",
    compute=compute,
    params=RadialMedianParams,
    kind="field",
    mode="both",
    needs=("sumimg", "real", "center"),
    doc="radial-median-subtracted residual z-score; needs beam center",
))
