"""
combine/union.py -- boolean-union combiner.

Unions the intensity-free floor with every per-detector thresholded pick. The
classic combiner; combine/weighted_sum.py is the continuous-fusion alternative.
"""

from __future__ import annotations
from dataclasses import dataclass

from automask.combine.base import CombineSpec, register_combine


@dataclass
class UnionParams:
    pass


def combine_masks(floor, picks):
    """Union the floor with every per-method pick (each already thresholded).
    `picks` is a dict[str, np.ndarray] for readability; the keys are not used."""
    combined = floor
    for m in picks.values():
        combined = combined | m
    return combined


def combine(floor, picks, sample, params: UnionParams | None = None):
    return combine_masks(floor, picks)


register_combine(
    CombineSpec(
        name="union",
        combine=combine,
        params=UnionParams,
        consumes="picks",
        doc="OR the floor with each thresholded per-detector pick",
    )
)
