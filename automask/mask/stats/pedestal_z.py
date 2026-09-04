"""
stats/pedestal_z.py -- per-ASIC median-polish robust-z of the pedestal constant.

A ``kind="field"`` statistic exposing the denoised pedestal as a per-pixel robust
z. The target defect (a leaky / high-dark-current patch) is a low-contrast filled
disk, below the per-pixel noise floor, so it is aggregated before thresholding:
pair with ``field_reg="blob_scale"`` and a high ``k``, then ``fill_holes`` +
``area_gate``.

Preprocessing: each ASIC has its additive row+column readout structure removed by
Tukey median polish, then is divided by its own residual MAD. ``status_as_mask``
bad pixels are excluded from every median and the MAD; a bad pixel's own z is 0.
Runs in native panel space and projects the z to the assembled canvas.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np

from automask.mask.stats.base import StatSpec, register_stat

MAD_TO_SIGMA = 1.4826


@dataclass
class PedestalZParams:
    asic: int = 256
    n_iter: int = 3
    k: float = 15.0  # threshold in post-blob_scale sigma
    mode: str = "high"


def masked_median_polish(block, bad, n_iter=3):
    """Tukey median polish ignoring the bad pixels, residual with NaN left at them."""
    r = block.astype(np.float64, copy=True)
    r[bad] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmedian(r)
        r -= 0.0 if np.isnan(g) else g
        for _ in range(n_iter):
            for axis in (1, 0):
                med = np.nanmedian(r, axis=axis, keepdims=True)
                r -= np.where(np.isnan(med), 0.0, med)
    return r


def residual_z(panel, p: PedestalZParams, bad=None):
    """Per-ASIC median-polish residual of the pedestal, as a robust z, native space.

    ``bad`` (native, True == masked) is excluded from every median and the MAD; a
    bad pixel's own z is set to 0."""
    panel = np.asarray(panel, dtype=np.float64)
    bad = np.zeros(panel.shape, bool) if bad is None else np.asarray(bad, bool)
    z = np.zeros_like(panel)
    for i in range(panel.shape[0]):
        for r0 in range(0, panel.shape[1], p.asic):
            for c0 in range(0, panel.shape[2], p.asic):
                sl = (i, slice(r0, r0 + p.asic), slice(c0, c0 + p.asic))
                m = bad[sl]
                res = masked_median_polish(panel[sl], m, p.n_iter)
                good = res[~m & np.isfinite(res)]
                sigma = (
                    MAD_TO_SIGMA * np.median(np.abs(good - np.median(good)))
                    if good.size
                    else 0.0
                )
                zz = res / (sigma + 1e-12)
                zz[~np.isfinite(zz)] = 0.0
                z[sl] = zz
    return z


def _bad_native(sample):
    """psana pixel status as a native bad-pixel mask (True == bad); psana 1 == good."""
    return np.asarray(sample.status_as_mask) == 0


def compute(sample, params: PedestalZParams | None = None):
    p = params or PedestalZParams()
    z = residual_z(sample.pedestals, p, _bad_native(sample))
    return sample.panel_to_asm(z)


register_stat(
    StatSpec(
        name="pedestal_z",
        compute=compute,
        params=PedestalZParams,
        kind="field",
        mode="high",
        needs=("pedestals", "status_as_mask"),
        doc="per-ASIC median-polish robust-z of the pedestal (bad pixels excluded); "
        "low-contrast defect, so pair with field_reg='blob_scale' + a high k",
    )
)
