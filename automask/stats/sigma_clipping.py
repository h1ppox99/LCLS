"""
stats/sigma_clipping.py -- azimuthal sigma-clip residual statistic (pyFAI-backed).

Identifies local outliers against the radially-symmetric scattering falloff via
pyFAI's iterative azimuthal sigma clipping (`AzimuthalIntegrator.sigma_clip_ng`
with ``error_model="azimuthal"``). At each iteration the 1D profile I(r) and the
per-ring azimuthal std are recomputed after discarding pixels with
``|I - <I>| > thres * std``, so bright/dead defects don't inflate the statistics
of their own ring -- unlike a plain radial mean/median subtraction. The converged
robust background I(r) and std(r) are projected back onto the detector with
`calcfrom1d`, and the evidence field is the per-pixel z-score
``(data - I(r)) / std(r)``: exactly the sigma-clip criterion, in sigma units.
Local outliers survive with large |z|; ring-following pixels cancel. Two-sided
(mode="both").

Geometry : Jungfrau1M, 75 um pixels, sample-detector 190 mm,
lambda 1.2915 A. The beam center (sample.center) sets the PONI; see
geometry.get_center for the axis-order subtlety.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat

PIXEL_M = 75e-6            # 75 um square pixels
DIST_M = 0.190            # sample-detector distance (not the stale psana z map)
WAVELENGTH_M = 1.2915e-10  # 1.2915 A
RADIAL_UNIT = "r_mm"      # bin on in-plane radial distance, matching the old px binning


@dataclass
class SigmaClippingParams:
    bin_width: float = 3.0      # px, radial annulus width
    thres: float = 5.0          # sigma-clip cut-off passed to sigma_clip_ng
    max_iter: int = 5           # sigma-clip iterations
    k: float = 5.0
    mode: str = "both"


def sigma_clipping_stat(sumimg, real, center, bin_width=3.0, thres=5.0, max_iter=5):
    """Per-pixel azimuthal sigma-clip z-score ``(data - I(r)) / std(r)`` via pyFAI.

    `center` is (axis0, axis1) in the frozen array's own index order
    (sample.center); it maps directly to pyFAI's (poni1, poni2). Masked pixels
    (~real) are excluded from the clipping and left at 0 in the output.
    `thres`/`max_iter` drive the iterative clipping that builds the robust I(r)
    and std(r); the returned z-score is then thresholded downstream at `k`.
    """
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    sumimg = sumimg.astype(np.float64, copy=False)
    c0, c1 = center
    ai = AzimuthalIntegrator(
        dist=DIST_M, poni1=c0 * PIXEL_M, poni2=c1 * PIXEL_M,
        pixel1=PIXEL_M, pixel2=PIXEL_M, wavelength=WAVELENGTH_M)
    ai.detector.shape = sumimg.shape
    mask = ~real

    # One radial bin every `bin_width` pixels, matching the old np.hypot binning.
    i0, i1 = np.indices(sumimg.shape)
    r_max_mm = float(np.hypot(i0 - c0, i1 - c1).max()) * PIXEL_M * 1000.0
    npt = max(1, int(r_max_mm / (bin_width * PIXEL_M * 1000.0)))

    # Iterative azimuthal sigma clip -> robust ring mean I(r) and per-ring std(r),
    # both computed after outliers have been discarded from each ring.
    res = ai.sigma_clip_ng(sumimg, npt=npt, mask=mask, unit=RADIAL_UNIT,
                           error_model="azimuthal", thres=thres, max_iter=max_iter)
    model = ai.calcfrom1d(res.radial, res.intensity, shape=sumimg.shape,
                          dim1_unit=RADIAL_UNIT, mask=mask)
    # res.std is the per-pixel azimuthal spread (what sigma_clip_ng thresholds
    # against); res.sigma/.sem is the standard error of the mean -- not this.
    std = ai.calcfrom1d(res.radial, res.std, shape=sumimg.shape,
                        dim1_unit=RADIAL_UNIT, mask=mask)

    z = np.zeros_like(sumimg)
    good = real & (std > 0)
    z[good] = (sumimg[good] - model[good]) / std[good]
    return z


def compute(sample, params: SigmaClippingParams | None = None):
    p = params or SigmaClippingParams()
    return sigma_clipping_stat(sample.sumimg, sample.real, sample.center,
                               p.bin_width, p.thres, p.max_iter)


register_stat(StatSpec(
    name="sigma_clipping",
    compute=compute,
    params=SigmaClippingParams,
    kind="field",
    mode="both",
    needs=("sumimg", "real", "center"),
    doc="azimuthal sigma-clip residual z-score (pyFAI sigma_clip_ng); needs beam center",
))
