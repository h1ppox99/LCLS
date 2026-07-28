"""
synthetic/artifacts.py -- deterministic synthetic artifact generators.

Two artifact classes are implemented: a straight Gaussian streak and a
beam-stop-like shadow (elliptical or rectangular). Each generator injects ONE
artifact into a source diffraction image and returns
``(corrupted_image, injected_mask)``:

    corrupted_image : float64, same shape as the source. The source is NEVER
                      modified in place, and originally-invalid pixels are left
                      exactly as they were.
    injected_mask   : bool, True where an artifact was injected -- the pixels a
                      masker is expected to flag. Always a subset of ``valid``.

Injection rule, enforced by every generator: only pixels marked valid in the
ground-truth mask (``valid = ~ground_truth``) are ever changed. Randomness comes
from a caller-supplied ``numpy.random.Generator``; a fixed seed reproduces the
corruption exactly. Geometry is expressed in image coordinates only -- no beam
center is required. Centres are given as fractions of the image extent so the
generators stay resolution-agnostic; widths/radii/lengths are in pixels.

Mask convention matches the rest of automask: True == masked/invalid.
"""
from __future__ import annotations

import numpy as np


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Median and a robust std (MAD-based) of a 1-D sample.

    MAD is used instead of np.std so a few real bright Bragg pixels in the
    source do not inflate the scale we build artifacts relative to.
    """
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    std = 1.4826 * mad
    if std == 0.0:                       # near-constant region -> fall back
        std = float(np.std(values)) or 1.0
    return med, std


def _as_pixel(center, shape) -> tuple[float, float]:
    """Fractional ``(row, col)`` in [0, 1] -> pixel coordinates."""
    fr, fc = center
    return fr * shape[0], fc * shape[1]


def _random_center(rng, shape) -> tuple[float, float]:
    """A random centre kept in the middle 40% of the frame (row, col px)."""
    return rng.uniform(0.3, 0.7) * shape[0], rng.uniform(0.3, 0.7) * shape[1]


# --------------------------------------------------------------------------
# 1. straight streak
# --------------------------------------------------------------------------
def streak_profile(shape, rng, *, center=None, angle_deg=None, width=2.0,
                   length=None, band_sigmas=1.0):
    """Geometry of a straight streak, independent of any intensity array.

    Returns ``(profile_unit, band)``: a UNIT-amplitude Gaussian line (times the
    on-segment gate) and the boolean within-``band_sigmas`` footprint. Callers
    scale ``profile_unit`` by an amplitude and add it to a real array. Splitting
    this out lets the single-image path and the Sample adapter build identical
    geometry from the same rng (draws: centre, then angle).
    """
    h, w = shape
    r0, c0 = _random_center(rng, shape) if center is None else _as_pixel(center, shape)
    if angle_deg is None:
        angle_deg = float(rng.uniform(0.0, 180.0))
    if length is None:
        length = float(np.hypot(h, w))              # cross the whole frame

    th = np.deg2rad(angle_deg)
    s, t = np.sin(th), np.cos(th)                   # unit direction (row, col)
    rr, cc = np.mgrid[0:h, 0:w]
    dr, dc = rr - r0, cc - c0
    perp = dr * t - dc * s                          # perpendicular distance
    along = dr * s + dc * t                         # distance along the line

    on_segment = np.abs(along) <= length / 2.0
    profile_unit = np.exp(-0.5 * (perp / width) ** 2) * on_segment
    band = (np.abs(perp) <= band_sigmas * width) & on_segment
    return profile_unit, band


def straight_streak(image, valid, rng, *, center=None, angle_deg=None,
                    width=2.0, amplitude_sigma=8.0, length=None,
                    band_sigmas=1.0):
    """Add a Gaussian-profile straight line (a scattering streak / zinger track).

    ``amplitude`` is set to ``amplitude_sigma`` robust std of the source, so the
    streak is bright relative to the local noise regardless of absolute counts.
    The injected footprint is the within-``band_sigmas`` core of the line (the
    faint Gaussian tails are added but not counted as ground-truth artifact).
    """
    out = np.array(image, dtype=np.float64, copy=True)
    profile_unit, band = streak_profile(
        image.shape, rng, center=center, angle_deg=angle_deg, width=width,
        length=length, band_sigmas=band_sigmas)
    _, sd = _robust_stats(out[valid])
    out[valid] += amplitude_sigma * sd * profile_unit[valid]   # invalid untouched
    injected = band & valid
    return out, injected


# --------------------------------------------------------------------------
# 2. beam-stop-like shadow (elliptical or rectangular)
# --------------------------------------------------------------------------
def beamstop_factor(shape, rng, *, center=None, radius=60.0, axis_ratio=1.0,
                    transmission=0.15, softness=0.15, shape_kind="random"):
    """Geometry of a beam-stop shadow, independent of any intensity array.

    Returns ``(factor, core)``: a multiplicative field (``transmission`` in the
    core, an optional soft ramp back to 1) and the boolean fully-shadowed core.
    ``shape_kind`` is "ellipse", "rect" or "random" (a per-example coin flip).
    ``a = radius`` is the row half-extent, ``b = radius * axis_ratio`` the column
    half-extent (draws: centre, then the shape coin flip when random).
    """
    h, w = shape
    r0, c0 = _random_center(rng, shape) if center is None else _as_pixel(center, shape)
    a, b = radius, radius * axis_ratio
    if shape_kind == "random":
        shape_kind = "rect" if rng.integers(2) else "ellipse"

    rr, cc = np.mgrid[0:h, 0:w]
    dr, dc = np.abs(rr - r0), np.abs(cc - c0)
    if shape_kind == "ellipse":
        norm = np.sqrt((dr / a) ** 2 + (dc / b) ** 2)
    elif shape_kind == "rect":
        norm = np.maximum(dr / a, dc / b)           # L-inf -> axis-aligned box
    else:
        raise ValueError(f"shape must be ellipse/rect/random, got {shape_kind!r}")

    factor = np.ones(shape)
    core = norm <= 1.0
    factor[core] = transmission
    if softness > 0:
        ramp = (norm > 1.0) & (norm <= 1.0 + softness)
        frac = (norm[ramp] - 1.0) / softness        # 0 at core edge -> 1
        factor[ramp] = transmission + frac * (1.0 - transmission)
    return factor, core


def beamstop_shadow(image, valid, rng, *, center=None, radius=60.0,
                    axis_ratio=1.0, transmission=0.15, softness=0.15,
                    shape="random"):
    """Multiply intensities inside an elliptical or rectangular region by
    ``transmission`` in (0, 1), with an optional soft boundary. The injected
    footprint is the fully-shadowed core (normalised distance <= 1)."""
    out = np.array(image, dtype=np.float64, copy=True)
    factor, core = beamstop_factor(
        image.shape, rng, center=center, radius=radius, axis_ratio=axis_ratio,
        transmission=transmission, softness=softness, shape_kind=shape)
    out[valid] = out[valid] * factor[valid]         # invalid pixels untouched
    injected = core & valid
    return out, injected


# --------------------------------------------------------------------------
# 3. hot patch -- additive bright compact region (dark-current / pedestal defect)
# --------------------------------------------------------------------------
def hot_patch(image, valid, rng, *, center=None, radius=20.0, axis_ratio=1.0,
              amplitude_sigma=8.0, softness=0.15, shape="random"):
    """Add a compact bright region: the polarity of a leaky / high-dark-current
    patch of sensor, as seen in the pedestal constants.

    Deliberately reuses ``beamstop_factor``'s geometry (ellipse or rectangle,
    ``axis_ratio`` controlling elongation) but ADDS instead of multiplying, so
    the two artifacts differ only in polarity and the shape sweep is shared.
    Sweeping ``axis_ratio`` is what measures how a compact-blob detector degrades
    on shapes that are not circles -- ``axis_ratio`` 1.0 is a disk, 0.6/1.6 are
    ellipses or boxes half again as long in one axis.

    ``amplitude_sigma`` is in robust-std units of the array being corrupted, so
    the same number means the same detectability across pedestal / intensity
    arrays. The injected footprint is the core (normalised distance <= 1).
    """
    out = np.array(image, dtype=np.float64, copy=True)
    profile, core = hot_patch_profile(
        image.shape, rng, center=center, radius=radius, axis_ratio=axis_ratio,
        softness=softness, shape_kind=shape)
    _, sd = _robust_stats(out[valid])
    out[valid] += amplitude_sigma * sd * profile[valid]      # invalid untouched
    return out, core & valid


def hot_patch_profile(shape, rng, *, center=None, radius=20.0, axis_ratio=1.0,
                      softness=0.15, shape_kind="random"):
    """Unit-amplitude additive profile for a hot patch: 1 in the core, ramping to
    0 through the soft boundary. Derived from ``beamstop_factor`` (transmission 0)
    so both artifacts draw identical geometry from the same rng state."""
    factor, core = beamstop_factor(
        shape, rng, center=center, radius=radius, axis_ratio=axis_ratio,
        transmission=0.0, softness=softness, shape_kind=shape_kind)
    return 1.0 - factor, core


# Registry: name -> generator. Order is the deterministic iteration order used
# by the evaluation loop (and therefore by the per-example seed derivation).
# `beamstop_small` reuses the same generator; the config gives it a smaller radius
# range (~half the extent -> ~1/4 the injected footprint) to cover small shadows.
ARTIFACTS = {
    "streak": straight_streak,
    "beamstop": beamstop_shadow,
    "beamstop_small": beamstop_shadow,
    "hot_patch": hot_patch,
}
