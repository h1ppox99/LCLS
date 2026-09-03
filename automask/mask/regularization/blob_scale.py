"""
regularization/blob_scale.py -- multi-scale matched filter for compact defects.

Field->field transform (applied before thresholding). Aggregates evidence that
is spatially coherent but too weak per pixel to threshold: at radius R, `z` is
convolved with a flat disk and divided by sqrt(N_R), so unit-variance noise
stays unit-variance (keeping `k` in sigma) while a coherent excess filling the
disk is amplified by sqrt(N_R) -- the matched filter for a disk-shaped signal.
Taking the max over a list of radii avoids having to know the defect size in
advance, at the cost of localization (the largest radius dilates the boundary).

Edge caveat: `mode="nearest"` correlates pixels within ~R of a border, so `k`
is effectively looser there; the `geometry` floor already masks panel borders,
covering most of that band.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from automask.mask.regularization.base import RegSpec, register_reg


@dataclass
class BlobScaleParams:
    # Disk radii in px -- the defect sizes to aggregate over. Detection is best
    # when a radius is comparable to the defect's own radius.
    radii: Sequence[float] = field(default_factory=lambda: (6, 9, 12, 16))


def _disk(radius: float) -> np.ndarray:
    r = int(round(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (np.hypot(yy, xx) <= radius).astype(np.float64)


def blob_scale(z, radii=(6, 9, 12, 16)):
    """Max over radii of the unit-variance-normalized disk average of `z`.

    Non-finite pixels are treated as 0 for the convolution and zeroed back out,
    so gaps neither leak signal nor drag neighbours down."""
    z = np.asarray(z, dtype=np.float64)
    if not len(radii):
        return z
    finite = np.isfinite(z)
    work = z if finite.all() else np.where(finite, z, 0.0)
    out = None
    for radius in radii:
        w = _disk(radius)
        # Dividing the disk SUM by sqrt(N) keeps unit variance under white noise
        # while scaling a coherent excess by sqrt(N).
        resp = ndi.convolve(work, w, mode="nearest") / np.sqrt(w.sum())
        out = resp if out is None else np.maximum(out, resp)
    return out if finite.all() else np.where(finite, out, 0.0)


def apply(z, params: BlobScaleParams | None = None):
    p = params or BlobScaleParams()
    return blob_scale(z, tuple(p.radii))


register_reg(
    RegSpec(
        name="blob_scale",
        apply=apply,
        params=BlobScaleParams,
        kind="field",
        doc="multi-scale disk matched filter; aggregates weak coherent evidence",
    )
)
