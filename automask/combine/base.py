"""
combine/base.py -- registry for combination modules.

A combiner fuses the per-detector evidence onto the intensity-free floor to
produce the final mask. Two families by what they consume:
  * consumes="picks"  -- a dict of already-thresholded boolean masks, unioned
    with the floor (combine/union.py).
  * consumes="fields" -- a dict of aligned continuous DEFECTIVENESS fields
    (large > 0 == wants masking), fused jointly then thresholded once
    (combine/weighted_sum.py, combine/mahalanobis.py).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Type

# name -> CombineSpec. Populated at import time by each combine module.
COMBINERS: Dict[str, "CombineSpec"] = {}


@dataclass
class CombineSpec:
    name: str
    combine: Callable  # (floor, components, sample, params) -> bool mask
    params: Type  # dataclass type holding this combiner's knob(s)
    consumes: str = "picks"  # "picks" or "fields"
    doc: str = ""


def register_combine(spec: CombineSpec) -> CombineSpec:
    COMBINERS[spec.name] = spec
    return spec
