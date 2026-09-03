"""
regularization/pad.py -- square-dilation mask regularizer (grows sparse picks).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from automask.mask.regularization.base import RegSpec, register_reg


@dataclass
class PadParams:
    pad: int = 2  # half-width; (2*pad+1) square dilation


def pad_mask(mask, pad=2):
    """Grow a boolean mask by `pad` px each side with a (2*pad+1) square dilation."""
    if not pad:
        return mask
    return ndi.binary_dilation(mask, structure=np.ones((2 * pad + 1, 2 * pad + 1)))


def apply(mask, params: PadParams | None = None):
    pad = 2 if params is None else params.pad
    return pad_mask(mask, pad)


register_reg(
    RegSpec(
        name="pad",
        apply=apply,
        params=PadParams,
        kind="mask",
        doc="square binary dilation to grow sparse thresholded picks",
    )
)
