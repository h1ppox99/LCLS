"""
combine/weighted_sum.py -- weighted-sum fusion combiner.

Instead of thresholding each statistic on its own and unioning the booleans, sum
the aligned defectiveness `fields` into ONE score S = sum_i w_i d_i and threshold
it a single time, then union the pick onto the intensity-free floor. The fields
are produced by the pipeline already sign-aligned (large > 0 == wants masking)
and on a comparable robust-z scale, so equal weights are the sensible default.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from automask.combine.base import CombineSpec, register_combine
from automask.regularization.pad import pad_mask


@dataclass
class WeightedSumParams:
    k: float = 3.5  # single threshold on the summed z-field
    pad: int = 2  # grow the sparse fused pick
    weights: Optional[List[float]] = None  # per-field weights; None -> equal


def combine_stats(floor, fields, real, weights=None, k=3.5, pad=2):
    """Weighted-sum fusion: S = sum_i w_i d_i, threshold once, pad, union on floor."""
    names = list(fields)
    w = np.ones(len(names)) if weights is None else np.asarray(weights, float)
    S = np.zeros(floor.shape, dtype=np.float64)
    for wi, n in zip(w, names):
        S += wi * fields[n]
    picked = pad_mask((S > k) & real, pad) & real
    return floor | picked


def combine(floor, fields, sample, params: WeightedSumParams | None = None):
    p = params or WeightedSumParams()
    return combine_stats(
        floor, fields, sample.real, weights=p.weights, k=p.k, pad=p.pad
    )


register_combine(
    CombineSpec(
        name="weighted_sum",
        combine=combine,
        params=WeightedSumParams,
        consumes="fields",
        doc="sum aligned defectiveness fields, threshold once, pad, union on floor",
    )
)
