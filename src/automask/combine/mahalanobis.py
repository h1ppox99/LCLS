"""
combine/mahalanobis.py -- robust joint-outlier fusion combiner.

Treat each pixel as a vector x = (d_var, d_wm, d_bh) of the aligned defectiveness
fields, fit a robust (hard-trimmed) mean mu + covariance Sigma over the good-pixel
bulk, and score every pixel by the Mahalanobis distance sqrt((x-mu)^T Sigma^-1
(x-mu)). Unlike weighted_sum this uses the field CORRELATIONS (a dead pixel is
low-variance AND dark). No labels needed, so it generalizes to label-free runs.
Promoted from studies/fusion_stats.py.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from combine.base import CombineSpec, register_combine
from regularization.pad import pad_mask


@dataclass
class MahalanobisParams:
    k: float = 3.0          # threshold on the Mahalanobis distance field
    pad: int = 2
    keep_frac: float = 0.8  # fraction of closest pixels kept when refitting
    iters: int = 5


def _trimmed_moments(X, keep_frac=0.8, iters=5):
    """Robust (hard-trimmed) mean + covariance a la a cheap MCD, numpy-only."""
    mu = np.median(X, axis=0)
    cov = np.cov(X, rowvar=False)
    for _ in range(iters):
        Xc = X - mu
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
        keep = d2 <= np.quantile(d2, keep_frac)
        mu = X[keep].mean(axis=0)
        cov = np.cov(X[keep], rowvar=False)
    return mu, cov


def mahalanobis_field(fields, domain, keep_frac=0.8, iters=5):
    """sqrt(D^2) per pixel over `domain`, 0 outside it."""
    X = np.column_stack([f[domain] for f in fields.values()])
    mu, cov = _trimmed_moments(X, keep_frac=keep_frac, iters=iters)
    inv = np.linalg.pinv(cov)
    Xc = X - mu
    d2 = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
    S = np.zeros(domain.shape, dtype=np.float64)
    S[domain] = np.sqrt(d2)
    return S


def combine(floor, fields, sample, params: MahalanobisParams | None = None):
    p = params or MahalanobisParams()
    real = sample.real
    S = mahalanobis_field(fields, real, keep_frac=p.keep_frac, iters=p.iters)
    picked = pad_mask((S > p.k) & real, p.pad) & real
    return floor | picked


register_combine(CombineSpec(
    name="mahalanobis",
    combine=combine,
    params=MahalanobisParams,
    consumes="fields",
    doc="robust Mahalanobis joint-outlier distance over the field stack",
))
