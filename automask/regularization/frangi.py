"""
regularization/frangi.py -- Frangi vesselness field regularizer.

A field->field transform (applied BEFORE thresholding, like `tv`): it runs the
multiscale Hessian-based Frangi filter over a continuous field and returns the
ridge/vesselness response. Where TV *denoises* a field, Frangi *enhances
line/tubular structure* -- so plugged in as a Detector.field_reg it turns a
stat field into a map that is large along curvilinear defects (scratches,
cracks, gap/ASIC lines) and ~0 elsewhere, which mode="high" thresholding then
picks out.

Knob (FrangiParams):
    sigmas       Hessian scales in px -- the ridge widths to enhance.
    beta         blob-vs-ridge sensitivity (Frangi's b; smaller = stricter ridge).
    gamma        structure-ness cutoff; None -> skimage's per-scale auto value.
    black_ridges False (default) enhances BRIGHT ridges -- the sign the pipeline's
                 defectiveness fields use (large == wants masking); True for dark.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from skimage.filters import frangi, threshold_minimum

from automask.regularization.base import RegSpec, register_reg


@dataclass
class FrangiParams:
    sigmas: Sequence[float] = field(default_factory=lambda: (1.0, 2.0, 3.0))
    beta: float = 0.5
    gamma: Optional[float] = None
    black_ridges: bool = False


def frangi_ridges(z, sigmas=(1.0, 2.0, 3.0), beta=0.5, gamma=None, black_ridges=False):
    """Frangi vesselness of a 2-D field. Non-finite pixels are treated as 0 for
    the Hessian and zeroed back out in the response, so gaps don't leak ridges."""
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    work = z if finite.all() else np.where(finite, z, 0.0)
    resp = frangi(
        work,
        sigmas=tuple(sigmas),
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode="reflect",
    )
    if not finite.all():
        resp = np.where(finite, resp, 0.0)
    return resp


def auto_threshold(resp, domain=None, floor=1e-9, nbins=256):
    """Unsupervised threshold for a Frangi vesselness response -- no ground truth.

    The response histogram is bimodal in LOG space: a broad noise-floor mode
    (~1e-9..1e-2) and a separated ridge mode (~1e-2..1). Otsu/triangle/Li/Yen
    fail here (they balance mass/variance and sink into the huge low mode);
    Prewitt's `minimum` method instead finds the VALLEY between the two peaks,
    which is exactly the detection cutoff. Computed on log10 of the positive
    responses over `domain`. Returns the response value k to use as `resp > k`.

    Verified run-agnostic on runs 389 & 475 (k~0.013, stable across nbins) and
    coincident with the supervised precision knee. Re-check the histogram is
    bimodal on very different runs before trusting it blindly."""
    v = resp if domain is None else resp[domain]
    v = v[v > floor]
    if v.size == 0:
        return float("inf")
    return float(10.0 ** threshold_minimum(np.log10(v), nbins=nbins))


def apply(z, params: FrangiParams | None = None):
    p = params if params is not None else FrangiParams()
    return frangi_ridges(
        z, sigmas=p.sigmas, beta=p.beta, gamma=p.gamma, black_ridges=p.black_ridges
    )


register_reg(
    RegSpec(
        name="frangi",
        apply=apply,
        params=FrangiParams,
        kind="field",
        doc="multiscale Frangi vesselness -- enhances line/ridge defects before thresholding",
    )
)
