"""
regularization/area_gate.py -- drop connected components below a minimum area.

A mask regularizer (mask -> mask, applied AFTER thresholding). Separates extended
defects from the scatter of small bright/dark specks that any threshold picks up:
on run 475 the pedestal defect is ~1200 px while the competing Bragg-spot
responses are 20-80 px, a 15x separation that area alone resolves cleanly.

Area is deliberately the only criterion. A circularity / eccentricity gate was
measured on the same data and made things worse -- it discards genuine defects
whose outline is irregular, which is most of them, and it re-introduces exactly
the shape prior that the detection stage is careful not to assume. Size is a
near-assumption-free property; shape is not.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from automask.regularization.base import RegSpec, register_reg


@dataclass
class AreaGateParams:
    min_area: int = 200      # components smaller than this are dropped
    connectivity: int = 2    # 1 = 4-connected, 2 = 8-connected


def area_gate(mask, min_area=200, connectivity=2):
    """Keep only connected components of at least `min_area` pixels."""
    mask = np.asarray(mask, dtype=bool)
    if min_area <= 1 or not mask.any():
        return mask
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask
    # Component 0 is the background; index areas by label to build a keep-LUT.
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def apply(mask, params: AreaGateParams | None = None):
    p = params or AreaGateParams()
    return area_gate(mask, p.min_area, p.connectivity)


register_reg(RegSpec(
    name="area_gate",
    apply=apply,
    params=AreaGateParams,
    kind="mask",
    doc="drop connected components below min_area (extended defects only)",
))
