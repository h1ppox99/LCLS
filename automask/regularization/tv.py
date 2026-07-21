"""
regularization/tv.py -- Total-Variation field regularizer (Chambolle).
"""
from __future__ import annotations
from dataclasses import dataclass

from skimage.restoration import denoise_tv_chambolle

from automask.regularization.base import RegSpec, register_reg


@dataclass
class TVParams:
    weight: float = 4.0     # data-fidelity <-> shape-prior knob; <=0 is a no-op


def tv_denoise(z, weight):
    """Isotropic Total-Variation denoising (Chambolle):
        u = argmin_u  ||u - z||^2 + weight * TV(u)
    weight <= 0 is a no-op passthrough."""
    return denoise_tv_chambolle(z, weight=weight) if weight > 0 else z


def apply(z, params: TVParams | None = None):
    weight = 4.0 if params is None else params.weight
    return tv_denoise(z, weight)


register_reg(RegSpec(
    name="tv",
    apply=apply,
    params=TVParams,
    kind="field",
    doc="isotropic TV denoising of the continuous field before thresholding",
))
