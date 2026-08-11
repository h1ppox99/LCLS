"""
stats/asic_polish.py -- per-ASIC median-polish residual on a calibration constant.

Detects dark-current / offset defects in the pedestal calibration constant
(not the scattering image) that psana's ``pixel_status`` does not flag. A
Jungfrau pedestal is dominated by readout structure -- column striping,
per-ASIC offsets -- larger than the defects sought, so within each 256x256
ASIC we remove that structure with Tukey's *median polish* (iterative row +
column median subtraction) and scale the residual by its own MAD for a robust
z. A compact defect barely perturbs those medians, same robustness principle
as ``sigma_clipping`` but via the median.

Useless alone: the per-pixel z of the targeted defect class sits below the
chip's own noise floor, so no threshold separates it. Pair with the
``blob_scale`` field regularizer, which aggregates the spatially coherent
evidence to make it separable.

Runs in native panel space (assembly scrambles ASIC boundaries) and projects
back to assembled space via the frozen index maps, so it stays numpy-only.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat

MAD_TO_SIGMA = 1.4826


@dataclass
class AsicPolishParams:
    asic: int = 256         # ASIC tile size (Jungfrau: 256x256)
    n_iter: int = 3         # median-polish sweeps (row+column per sweep)
    k: float = 15.0         # threshold, in post-aggregation sigma (see blob_scale)
    mode: str = "high"      # flag pixels ABOVE the local pedestal (hot/leaky)


def median_polish(block, n_iter=3):
    """Tukey median polish of a 2-D block: remove the additive row+column fit.

    Returns the residual. Robust to a compact anomaly covering a minority of the
    block's rows and columns, which is exactly the defect class this targets."""
    r = block.astype(np.float64, copy=True)
    r -= np.median(r)
    for _ in range(n_iter):
        r -= np.median(r, axis=1, keepdims=True)
        r -= np.median(r, axis=0, keepdims=True)
    return r


def asic_polish_stat(panel, run, asic=256, n_iter=3):
    """Per-ASIC median-polish residual of `panel`, as a robust z, in ASSEMBLED space.

    `panel` is a native (2, 512, 1024) calibration constant; `run` selects the
    frozen panel->assembled index maps. Each ASIC is scaled by its own residual
    MAD, so a noisier ASIC does not dominate the chip-wide field."""
    from automask.geometry import panel_to_asm

    panel = np.asarray(panel, dtype=np.float64)
    z = np.zeros_like(panel)
    for p in range(panel.shape[0]):
        for r0 in range(0, panel.shape[1], asic):
            for c0 in range(0, panel.shape[2], asic):
                res = median_polish(panel[p, r0:r0+asic, c0:c0+asic], n_iter)
                sigma = MAD_TO_SIGMA * np.median(np.abs(res - np.median(res)))
                z[p, r0:r0+asic, c0:c0+asic] = res / (sigma + 1e-12)
    return panel_to_asm(z, run)


def compute(sample, params: AsicPolishParams | None = None):
    p = params or AsicPolishParams()
    if sample.pedestals is None:
        raise ValueError(
            "asic_polish needs the 'pedestals' calibration; load_sample was called "
            "without it (see Pipeline.calibrations_needed)")
    return asic_polish_stat(sample.pedestals, sample.run, p.asic, p.n_iter)


register_stat(StatSpec(
    name="asic_polish",
    compute=compute,
    params=AsicPolishParams,
    kind="field",
    mode="high",
    needs=("pedestals",),
    doc="per-ASIC median-polish z of the pedestal constants; pair with blob_scale",
))
