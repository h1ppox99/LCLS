"""
regularization/blob_scale.py -- multi-scale matched filter for compact defects.

Field->field transform (applied before thresholding). Aggregates evidence that
is spatially coherent but too weak per pixel to threshold: `z` is convolved with
a flat kernel and divided by sqrt(N), so unit-variance noise stays unit-variance
(keeping `k` in sigma) while a coherent excess filling the kernel is amplified by
sqrt(N) -- the matched filter for that kernel shape. The max is taken over a bank
of kernels so nothing has to know the defect size or shape in advance.

The kernel is an ellipse of semi-axes (radius, radius*aspect) at `angle`; the
default `aspects=(1,)` makes it a disk, reproducing the isotropic filter. Adding
`aspects`/`angles` extends the bank to elongated shapes. Because the bank is
combined by `max`, a wider bank can only raise a response, never suppress one --
it improves recall for elongated defects without risking a false negative, at a
mild precision cost (a larger bank inflates the noise tail, read off the
distribution when choosing `k`).

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
    # radii: defect sizes to aggregate over (a radius near the defect's own is best).
    # aspects/angles extend the bank to elongated shapes; the defaults are a disk.
    radii: Sequence[float] = field(default_factory=lambda: (6, 9, 12, 16))
    aspects: Sequence[float] = field(default_factory=lambda: (1.0,))
    angles: Sequence[float] = field(default_factory=lambda: (0.0,))


def _disk(radius: float) -> np.ndarray:
    r = int(round(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (np.hypot(yy, xx) <= radius).astype(np.float64)


def _ellipse(radius: float, aspect: float = 1.0, angle: float = 0.0) -> np.ndarray:
    a, b = radius, radius * aspect
    r = int(round(max(a, b)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    th = np.deg2rad(angle)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    return ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0).astype(np.float64)


def blob_scale(z, radii=(6, 9, 12, 16), aspects=(1.0,), angles=(0.0,)):
    """Max over a kernel bank of the unit-variance-normalized flat average of `z`.

    Non-finite pixels are treated as 0 for the convolution and zeroed back out,
    so gaps neither leak signal nor drag neighbours down."""
    z = np.asarray(z, dtype=np.float64)
    if not len(radii):
        return z
    finite = np.isfinite(z)
    work = z if finite.all() else np.where(finite, z, 0.0)
    out = None
    for radius in radii:
        for aspect in aspects:
            for angle in angles if aspect != 1.0 else (0.0,):
                w = _ellipse(radius, aspect, angle)
                resp = ndi.convolve(work, w, mode="nearest") / np.sqrt(w.sum())
                out = resp if out is None else np.maximum(out, resp)
    return out if finite.all() else np.where(finite, out, 0.0)


def apply(z, params: BlobScaleParams | None = None):
    p = params or BlobScaleParams()
    return blob_scale(z, tuple(p.radii), tuple(p.aspects), tuple(p.angles))


register_reg(
    RegSpec(
        name="blob_scale",
        apply=apply,
        params=BlobScaleParams,
        kind="field",
        doc="multi-scale oriented matched filter (disk by default); aggregates weak "
        "coherent evidence; add aspects/angles for elongated defects",
    )
)
