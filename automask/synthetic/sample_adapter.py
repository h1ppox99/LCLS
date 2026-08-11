"""
synthetic/sample_adapter.py -- inject synthetic artifacts into a run-level Sample.

The single-image path (evaluate.py, mode="image") scores a masker that sees only
one 2-D image, so its variance channel is absent. This adapter instead corrupts a
full ``automask.sample.Sample`` so the *production* ``Pipeline`` -- variance
(reads ``std``), blackhat (reads ``mean``), geometry + pixel-status floor
-- can be scored on the same synthetic artifacts.

An artifact is injected CONSISTENTLY across the intensity-derived arrays a real
artifact would move together:

  * streak   -- additive bright line. Added to ``mean``, scaled to its robust
                std (the amplitude is in sigma units).
                ``std`` is left unchanged: extra photons would only *raise*
                shot noise, and the low-variance detector (mode="low") keys on
                DARK/dead pixels -- a bright streak is out of its polarity, which
                the score will (correctly) reflect.
  * beamstop -- multiplicative shadow. ``mean`` AND ``std`` are both
                multiplied by the same transmission factor (a simple linear
                attenuation model): fewer counts -> lower mean AND lower spread,
                so the shadow shows up as a low-variance, locally-dark island the
                production pipeline is built to catch.

Geometry stays valid because the factor is > 0 (real pixels stay
real) and the streak is additive. Only pixels in the caller-supplied valid
``region`` are modified, and the injected mask is a subset of it, matching the
image-path convention. The source Sample is never mutated.
"""
from __future__ import annotations

import numpy as np

from automask.sample import Sample
from automask.synthetic.artifacts import (beamstop_factor, hot_patch_profile,
                                          streak_profile, _robust_stats)


def rotate_sample(sample, degrees: int):
    """Rotate every array field of a Sample together by a multiple of 90 deg.

    Keeps the arrays mutually aligned (90/270 also transpose the shape). ``run``
    is preserved; the production detectors used here (variance/blackhat +
    geometry/calib floor) read no absolute geometry, so the stale
    ``center`` cached-property is irrelevant.

    Panel-form fields (``pedestals``, ``pixel_rms``) are DROPPED to None under a
    non-zero rotation. They live in native (2, 512, 1024) geometry, which has no
    meaningful image rotation, and the frozen ix/iy maps relating them to
    assembled space would no longer apply. Leaving them unrotated would misalign
    them against every other array with no error, which is the one outcome to
    avoid; dropping them instead makes any consumer fail loudly at the point of
    use. Cases that need these constants therefore declare ``rotations: [0]``.
    """
    if int(degrees) % 90 != 0:
        raise ValueError(f"rotations must be multiples of 90, got {degrees}")
    k = (int(degrees) // 90) % 4
    arrays = {}
    for name, array in sample.arrays.items():
        if array.ndim == 3:          # native panel geometry, not assembled
            if k:
                continue             # dropped: see the note above
            arrays[name] = array
        else:
            arrays[name] = np.rot90(array, k)
    return Sample(run=sample.run, arrays=arrays, selection=sample.selection)


def corrupt_sample(sample, name: str, rng, params: dict, region):
    """Inject one artifact into ``sample``; return ``(corrupted_sample, injected)``.

    ``params`` are the already-resolved generator parameters (as produced by the
    evaluate.py loop). ``region`` is the boolean map of originally-valid pixels
    the artifact may touch -- supplied by the caller rather than read off the
    Sample, which no longer carries a reference mask. Geometry is built from
    ``rng`` exactly as the single-image generators build it, so a given seed
    reproduces the same artifact in either mode. ``injected`` is bool, True where
    the artifact was placed (subset of ``region``).
    """
    grid = sample.mean.shape
    p = dict(params)

    if name == "streak":
        amplitude_sigma = p.pop("amplitude_sigma", 8.0)
        profile_unit, band = streak_profile(grid, rng, **p)
        updates = {}
        for field in ("mean",):
            arr = np.array(getattr(sample, field), dtype=np.float64, copy=True)
            _, sd = _robust_stats(arr[region])
            arr[region] += amplitude_sigma * sd * profile_unit[region]
            updates[field] = arr
        injected = band & region

    elif name in ("beamstop", "beamstop_small"):
        p.setdefault("shape_kind", p.pop("shape", "random"))
        factor, core = beamstop_factor(grid, rng, **p)
        updates = {}
        for field in ("mean", "std"):
            arr = np.array(getattr(sample, field), dtype=np.float64, copy=True)
            arr[region] = arr[region] * factor[region]
            updates[field] = arr
        injected = core & region

    elif name == "hot_patch":
        # Additive bright patch in the PEDESTAL constants -- a leaky/high-dark-
        # current region of sensor. Geometry is drawn in assembled space (so the
        # injected mask is comparable with the other artifacts and the reference)
        # and mapped into native panel geometry to corrupt the constant itself.
        from automask.geometry import asm_to_panel

        p.setdefault("shape_kind", p.pop("shape", "random"))
        amplitude_sigma = p.pop("amplitude_sigma", 8.0)
        profile, core = hot_patch_profile(grid, rng, **p)
        ped = np.array(sample.pedestals, dtype=np.float64, copy=True)
        valid_panel = asm_to_panel(region, sample.run)
        _, sd = _robust_stats(ped[valid_panel])
        ped[valid_panel] += (amplitude_sigma * sd
                             * asm_to_panel(profile, sample.run)[valid_panel])
        updates = {"pedestals": ped}
        injected = core & region

    else:
        raise ValueError(f"no Sample adapter for artifact {name!r} "
                         f"(supported: streak, beamstop, beamstop_small, hot_patch)")

    return sample.with_arrays(**updates), injected
