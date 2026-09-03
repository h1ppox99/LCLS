"""
regularization/fill_holes.py -- close interior holes in a boolean mask.

A mask regularizer (mask -> mask, applied AFTER thresholding). A threshold on a
graded field leaves pinholes wherever the evidence dips below `k` inside an
otherwise solid defect; those pixels are part of the defect, so filling them is a
correction rather than padding slack. Unlike `pad`, this grows nothing at the
outer boundary -- only fully enclosed background is turned on.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from automask.mask.regularization.base import RegSpec, register_reg


@dataclass
class FillHolesParams:
    connectivity: int = 1  # background connectivity; 1 (4-connected) fills more


def fill_holes(mask, connectivity=1):
    """Fill fully enclosed background regions of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    return ndi.binary_fill_holes(mask, structure=structure)


def apply(mask, params: FillHolesParams | None = None):
    p = params or FillHolesParams()
    return fill_holes(mask, p.connectivity)


register_reg(
    RegSpec(
        name="fill_holes",
        apply=apply,
        params=FillHolesParams,
        kind="mask",
        doc="fill enclosed holes left by thresholding a graded field",
    )
)
