"""
stats/azimuthal_sigma.py -- azimuthal sigma-clipping statistic (promoted).

Per-pixel z-score from a robust azimuthal (ring) profile: bin every pixel by its
q, iteratively sigma-clip each q-bin's mean/std, then z-score every pixel against
its bin. Outliers (hot/dead pixels, panel artifacts, beam-stop edge) that deviate
from the ring get large |z|. Two-sided (mode="both").

This is the numpy-only hand-rolled sigma-clip from studies/pyfai_sigma_clip.py
(equivalent to pyFAI's ai.sigma_clip_ng), so the pipeline needs no pyFAI. Geometry
constants are the fixed xppl1016922 values (CLAUDE.md).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from stats.base import StatSpec, register_stat

DIST_M = 0.190              # sample-detector distance
WAVELENGTH_M = 1.2915e-10   # 9.6 keV
PIXEL_M = 75e-6


def _q_map(shape, c0, c1):
    """q [nm^-1] at every assembled pixel from pinhole-camera geometry."""
    i0, i1 = np.indices(shape)
    r = np.hypot((i0 - c0) * PIXEL_M, (i1 - c1) * PIXEL_M)
    two_theta = np.arctan2(r, DIST_M)
    return (4 * np.pi / WAVELENGTH_M * np.sin(two_theta / 2)) * 1e-9


@dataclass
class AzimuthalSigmaParams:
    npt: int = 1024             # number of q-bins
    clip_k: float = 5.0         # sigma-clip rejection threshold during the fit
    k: float = 5.0              # final |z| mask threshold
    mode: str = "both"
    max_iter: int = 5


def azimuthal_sigma_stat(sumimg, real, center, npt=1024, clip_k=5.0, max_iter=5):
    """Per-pixel z against an iteratively sigma-clipped azimuthal profile.
    Operates on the lit-beam run-sum `sumimg` over valid pixels `real`."""
    sumimg = sumimg.astype(np.float64, copy=False)
    qmap = _q_map(sumimg.shape, *center)
    cand = real
    edges = np.linspace(qmap[cand].min(), qmap[cand].max() + 1e-9, npt + 1)
    bidx = np.clip(np.digitize(qmap, edges) - 1, 0, npt - 1)

    included = cand.copy()
    mu_px = np.zeros_like(sumimg)
    std_px = np.zeros_like(sumimg)
    for _ in range(max_iter):
        cnt = np.bincount(bidx[included], minlength=npt).astype(np.float64)
        sm = np.bincount(bidx[included], weights=sumimg[included], minlength=npt)
        sq = np.bincount(bidx[included], weights=sumimg[included] ** 2, minlength=npt)
        mu_bin = np.divide(sm, cnt, out=np.zeros(npt), where=cnt > 0)
        var_bin = np.divide(sq, cnt, out=np.zeros(npt), where=cnt > 0) - mu_bin ** 2
        std_bin = np.sqrt(np.clip(var_bin, 0, None))
        mu_px, std_px = mu_bin[bidx], std_bin[bidx]
        new_included = cand & (np.abs(sumimg - mu_px) <= clip_k * std_px)
        if np.array_equal(new_included, included):
            break
        included = new_included

    z = np.zeros_like(sumimg)
    ok = cand & (std_px > 0)
    z[ok] = (sumimg[ok] - mu_px[ok]) / std_px[ok]
    return z


def compute(sample, params: AzimuthalSigmaParams | None = None):
    p = params or AzimuthalSigmaParams()
    return azimuthal_sigma_stat(sample.sumimg, sample.real, sample.center,
                                npt=p.npt, clip_k=p.clip_k, max_iter=p.max_iter)


register_stat(StatSpec(
    name="azimuthal_sigma",
    compute=compute,
    params=AzimuthalSigmaParams,
    kind="field",
    mode="both",
    needs=("sumimg", "real", "center"),
    doc="per-pixel z vs a sigma-clipped azimuthal profile; needs beam center",
))
