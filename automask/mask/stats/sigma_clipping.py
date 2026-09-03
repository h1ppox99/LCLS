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

POLARIZATION IS PART OF THE MODEL here, not an optional refinement: this
statistic assumes I depends on r alone, and polarization is the only
deterministic correction that breaks that -- a cos(2*chi) modulation at fixed r,
4.6% rms within a ring at q ~ 1.7. Uncorrected it lands in `std(r)`, the
z-score's own denominator, and caps what the statistic can see. On a synthetic
isotropic ring stack with a -20% shadow at q = 1.54 it costs the detection
outright: <z> = -4.1 with 7% of the patch over k=5, against <z> = -20 and all of
it once corrected. It also leaves a two-lobed pattern in the evidence field
(corr(z, cos 2chi) = +0.96) that the pipeline reads as real. It is applied on the
way in and re-applied on the way out, so `model`/`std` come back in `image`'s
raw units. Solid angle needs no such care -- a function of 2theta alone
(within-ring spread 1.2e-3), so the ring model absorbs it.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.mask.stats.base import StatSpec, register_stat

PIXEL_M = 75e-6  # 75 um square pixels
DIST_M = 0.190  # sample-detector distance (not the stale psana z map)
WAVELENGTH_M = 1.2915e-10  # 1.2915 A
RADIAL_UNIT = "r_mm"  # bin on in-plane radial distance, matching the old px binning

# pyFAI's polarization_factor is signed by which detector axis the E-field lies
# along: +1 = axis2 (fast), -1 = axis1 (slow). Our assembled arrays are
# transposed (axis0 = ix = lab x), so the horizontal LCLS polarization is -1
# here. Verified, not assumed: at -1 `ai.polarization()` reproduces psana's own
# `azav__azav_pol` array to 1e-4 over all 1M pixels, +1 anti-correlates.
POLARIZATION = -1.0


@dataclass
class SigmaClippingParams:
    bin_width: float = 3.0  # px, radial annulus width
    thres: float = 5.0  # sigma-clip cut-off passed to sigma_clip_ng
    max_iter: int = 5  # sigma-clip iterations
    k: float = 5.0
    mode: str = "both"


def sigma_clipping_stat(image, real, center, bin_width=3.0, thres=5.0, max_iter=5):
    """Per-pixel azimuthal sigma-clip z-score ``(data - I(r)) / std(r)`` via pyFAI.

    `center` is (axis0, axis1) in the frozen array's own index order
    (sample.center); it maps directly to pyFAI's (poni1, poni2). Masked pixels
    (~real) are excluded from the clipping and left at 0 in the output.
    `thres`/`max_iter` drive the iterative clipping that builds the robust I(r)
    and std(r); the returned z-score is then thresholded downstream at `k`.
    """
    # Deferred so numpy-only stats do not import pyFAI until needed.
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    image = image.astype(np.float64, copy=False)
    c0, c1 = center
    # +0.5: PONI is measured from the CORNER of pixel [0,0] while `center` is a
    # pixel INDEX, whose centre sits half a pixel further out. Verified against
    # psana's own q map (rms 7e-5 vs 4.9e-4 A^-1).
    ai = AzimuthalIntegrator(
        dist=DIST_M,
        poni1=(c0 + 0.5) * PIXEL_M,
        poni2=(c1 + 0.5) * PIXEL_M,
        pixel1=PIXEL_M,
        pixel2=PIXEL_M,
        wavelength=WAVELENGTH_M,
    )
    ai.detector.shape = image.shape
    mask = ~real

    # One radial bin every `bin_width` pixels, matching the old np.hypot binning.
    i0, i1 = np.indices(image.shape)
    r_max_mm = float(np.hypot(i0 - c0, i1 - c1).max()) * PIXEL_M * 1000.0
    npt = max(1, int(r_max_mm / (bin_width * PIXEL_M * 1000.0)))

    # Iterative azimuthal sigma clip -> robust ring mean I(r) and per-ring std(r),
    # both computed after outliers have been discarded from each ring.
    res = ai.sigma_clip_ng(
        image,
        npt=npt,
        mask=mask,
        unit=RADIAL_UNIT,
        error_model="azimuthal",
        thres=thres,
        max_iter=max_iter,
        polarization_factor=POLARIZATION,
    )
    model = ai.calcfrom1d(
        res.radial,
        res.intensity,
        shape=image.shape,
        dim1_unit=RADIAL_UNIT,
        mask=mask,
        polarization_factor=POLARIZATION,
    )
    # res.std is the per-pixel azimuthal spread (what sigma_clip_ng thresholds
    # against); res.sigma/.sem is the standard error of the mean -- not this. It
    # is a spread in corrected space, so it takes the same factor back as `model`.
    std = ai.calcfrom1d(
        res.radial,
        res.std,
        shape=image.shape,
        dim1_unit=RADIAL_UNIT,
        mask=mask,
        polarization_factor=POLARIZATION,
    )

    z = np.zeros_like(image)
    good = real & (std > 0)
    z[good] = (image[good] - model[good]) / std[good]
    return z


def compute(sample, params: SigmaClippingParams | None = None):
    p = params or SigmaClippingParams()
    return sigma_clipping_stat(
        sample.mean, sample.real, sample.center, p.bin_width, p.thres, p.max_iter
    )


register_stat(
    StatSpec(
        name="sigma_clipping",
        compute=compute,
        params=SigmaClippingParams,
        kind="field",
        mode="both",
        needs=("mean", "real", "center"),
        doc="azimuthal sigma-clip residual z-score (pyFAI sigma_clip_ng); needs beam center",
    )
)
