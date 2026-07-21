"""
regularization/close_open.py -- morphological close-then-open mask regularizer.

Legacy baseline superseded by TV; kept so studies/ can compare against it. Dense
masks only.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from automask.regularization.base import RegSpec, register_reg


@dataclass
class CloseOpenParams:
    size: int = 9           # square structuring-element size


def close_open(mask, size=9):
    """Binary close then open with a (size, size) square. Superseded by TV; kept
    for the studies/ that compare against it. Dense masks only."""
    se = np.ones((size, size), bool)
    return ndi.binary_opening(ndi.binary_closing(mask, structure=se), structure=se)


def apply(mask, params: CloseOpenParams | None = None):
    size = 9 if params is None else params.size
    return close_open(mask, size)


register_reg(RegSpec(
    name="close_open",
    apply=apply,
    params=CloseOpenParams,
    kind="mask",
    doc="binary close-then-open (legacy morphology baseline)",
))
