"""Detector bad-pixel floor, from psana's per-run pixel status."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from automask.mask.regularization.pad import pad_mask
from automask.mask.stats.base import StatSpec, register_stat


@dataclass
class StatusAsMaskParams:
    """`pad` widens each flagged pixel by `pad` px per side with a (2*pad+1)
    square dilation, matching the lab baseline's 5x5 dead-pixel dilation (pad=2)
    and the geometry floor stat. A dead pixel corrupts its neighbours (charge
    sharing / interpolation), so the halo is genuinely bad, not padding slack."""

    pad: int = 2


def compute(sample, params: StatusAsMaskParams | None = None):
    """psana's pixel status for this run, as a mask (``True == masked``).

    The array arrives in native panel form and in psana's convention, where
    ``1 == good``; both are converted here rather than at the store, so the
    unmapped canvas stays unmasked instead of inheriting a fill value.
    """
    p = params or StatusAsMaskParams()
    bad = np.asarray(sample.status_as_mask) == 0
    return pad_mask(sample.panel_to_asm(bad), p.pad)


register_stat(
    StatSpec(
        name="status_as_mask",
        compute=compute,
        params=StatusAsMaskParams,
        kind="floor",
        needs=("status_as_mask",),
        doc="psana pixel-status bad-pixel mask, dilated; intensity-free floor",
    )
)
