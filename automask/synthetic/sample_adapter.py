"""
synthetic/sample_adapter.py -- inject synthetic artifacts into a run-level Sample.

The single-image path (evaluate.py, mode="image") scores a masker that sees only
one 2-D image, so its variance channel is absent. This adapter instead corrupts a
full ``automask.evaluation.Sample`` so the *production* ``Pipeline`` -- variance
(reads ``ustd``), window_median + blackhat (read ``umean``), geometry+calib floor
-- can be scored on the same synthetic artifacts.

An artifact is injected CONSISTENTLY across the intensity-derived arrays a real
artifact would move together:

  * streak   -- additive bright line. Added to ``sumimg`` and ``umean``, each
                scaled to its OWN robust std (the amplitude is in sigma units).
                ``ustd`` is left unchanged: extra photons would only *raise*
                shot noise, and the low-variance detector (mode="low") keys on
                DARK/dead pixels -- a bright streak is out of its polarity, which
                the score will (correctly) reflect.
  * beamstop -- multiplicative shadow. ``sumimg``, ``umean`` AND ``ustd`` are all
                multiplied by the same transmission factor (a simple linear
                attenuation model): fewer counts -> lower mean AND lower spread,
                so the shadow shows up as a low-variance, locally-dark island the
                production pipeline is built to catch.

``mean`` (dark frame), ``human`` and ``calib`` are untouched -- no production stat
moves them. Geometry stays valid because the factor is > 0 (real pixels stay
real) and the streak is additive. Only originally-valid pixels (``~human``) are
modified, and the injected mask is a subset of that region, matching the
image-path convention. The source Sample is never mutated.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from automask.synthetic.artifacts import beamstop_factor, streak_profile, _robust_stats


def rotate_sample(sample, degrees: int):
    """Rotate every array field of a Sample together by a multiple of 90 deg.

    Keeps the arrays mutually aligned (90/270 also transpose the shape). ``run``
    is preserved; the production detectors used here (variance/window_median/
    blackhat + geometry/calib floor) read no absolute geometry, so the stale
    ``center`` cached-property is irrelevant.
    """
    if int(degrees) % 90 != 0:
        raise ValueError(f"rotations must be multiples of 90, got {degrees}")
    k = (int(degrees) // 90) % 4
    rot = lambda a: np.rot90(a, k)
    return replace(sample, sumimg=rot(sample.sumimg), mean=rot(sample.mean),
                   umean=rot(sample.umean), ustd=rot(sample.ustd),
                   human=rot(sample.human), calib=rot(sample.calib))


def corrupt_sample(sample, name: str, rng, params: dict):
    """Inject one artifact into ``sample``; return ``(corrupted_sample, injected)``.

    ``params`` are the already-resolved generator parameters (as produced by the
    evaluate.py loop). Geometry is built from ``rng`` exactly as the single-image
    generators build it, so a given seed reproduces the same artifact in either
    mode. ``injected`` is bool, True where the artifact was placed (subset of the
    originally-valid region).
    """
    region = ~sample.human                          # originally-valid pixels
    grid = sample.sumimg.shape
    p = dict(params)

    if name == "streak":
        amplitude_sigma = p.pop("amplitude_sigma", 8.0)
        profile_unit, band = streak_profile(grid, rng, **p)
        updates = {}
        for field in ("sumimg", "umean"):
            arr = np.array(getattr(sample, field), dtype=np.float64, copy=True)
            _, sd = _robust_stats(arr[region])
            arr[region] += amplitude_sigma * sd * profile_unit[region]
            updates[field] = arr
        injected = band & region

    elif name in ("beamstop", "beamstop_small"):
        p.setdefault("shape_kind", p.pop("shape", "random"))
        factor, core = beamstop_factor(grid, rng, **p)
        updates = {}
        for field in ("sumimg", "umean", "ustd"):
            arr = np.array(getattr(sample, field), dtype=np.float64, copy=True)
            arr[region] = arr[region] * factor[region]
            updates[field] = arr
        injected = core & region

    else:
        raise ValueError(f"no Sample adapter for artifact {name!r} "
                         f"(supported: streak, beamstop, beamstop_small)")

    return replace(sample, **updates), injected
