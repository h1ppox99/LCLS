"""
regularization/base.py -- registry for regularization modules.

A regularizer smooths/cleans one stage of the pipeline under a single tunable
knob, replacing ad-hoc morphology. Two kinds by what they operate on:
  * KIND="field" -- continuous field -> continuous field (e.g. TV denoising),
    applied BEFORE thresholding.
  * KIND="mask"  -- boolean mask -> boolean mask (e.g. padding / close-open),
    applied AFTER thresholding.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Type

# name -> RegSpec. Populated at import time by each regularization module.
REGULARIZERS: Dict[str, "RegSpec"] = {}


@dataclass
class RegSpec:
    name: str
    apply: Callable        # (x, params) -> x  (field->field or mask->mask)
    params: Type           # dataclass type holding this regularizer's knob(s)
    kind: str = "field"    # "field" or "mask"
    doc: str = ""


def register_reg(spec: RegSpec) -> RegSpec:
    REGULARIZERS[spec.name] = spec
    return spec
