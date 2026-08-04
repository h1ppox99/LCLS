"""
identification/priors.py -- one MAP detector per artifact class, acting on
evidence rather than on a mask.

The claim these implement: `argmax_M sum_i l_i(M_i) - sum_k Phi_k(M_k)` with a
DIFFERENT `Phi_k` per artifact class, because the three classes have
incompatible geometry and a single shape prior is not merely suboptimal for two
of them, it has the wrong sign.

    shadow  compact region, sharp edge   Phi_reg = lambda * |boundary of M|
    streak  thickened curve              Phi_curve = lambda_w * w + lambda_n
    blob    disk of unknown radius       matched-filter bank, max_R l_R(i)

The sharp version of the argument is about the perimeter prior on a thin line:
a `L x w` streak has perimeter `~2L`, which grows with its length, so
`Phi_reg` charges a streak in proportion to how much evidence it carries. Any
`lambda` large enough to suppress an isolated noisy pixel is large enough to
delete a real streak. `Phi_curve` charges width and count instead, and is flat
in length. `exp_p1_prior_mismatch` measures the resulting phase diagram.

WHY THE PERIMETER MAP IS EXACT AND WHY THAT MATTERS. `Phi_reg` is submodular, so
the binary MAP is a graph cut with a global optimum. This module gets that
optimum without a max-flow dependency, through the level-set property of
Rudin-Osher-Fatemi denoising (Chambolle-Darbon): the super-level set `{u > t}`
of `argmin_u 1/2||u - f||^2 + lambda TV(u)` is a global minimizer of
`lambda Per(S) + int_S (t - f)`. Running ROF on the evidence field and
thresholding at zero is therefore the exact MAP for
`max_S int_S l - lambda Per(S)`, computed in numpy.

That exactness is the whole point of the comparison against the current
pipeline. `masking.Pipeline` thresholds a statistic and THEN applies morphology,
so the shape prior only ever sees a binary array -- the evidence that a pixel
was marginal, which is exactly what a shape prior needs in order to rescue or
discard it, has already been destroyed. Whether that costs anything measurable
is `exp_o1_greedy_vs_joint`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


# ==========================================================================
#  shadow -- exact MAP under a perimeter prior
# ==========================================================================
def rof(f: np.ndarray, lam: float, n_iter: int = 200,
        step: float = 0.24) -> np.ndarray:
    """Chambolle's projected-dual solution of `min_u 1/2||u-f||^2 + lam TV(u)`.

    `step <= 1/4` is the stability bound for the 4-connected discrete gradient.
    """
    f = np.asarray(f, dtype=np.float64)
    if lam <= 0:
        return f.copy()
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    for _ in range(n_iter):
        div = np.zeros_like(f)
        div[:-1, :] += px[:-1, :]
        div[1:, :] -= px[:-1, :]
        div[:, :-1] += py[:, :-1]
        div[:, 1:] -= py[:, :-1]
        u = div - f / lam
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]
        norm = np.sqrt(gx ** 2 + gy ** 2)
        px = (px + step * gx) / (1.0 + step * norm)
        py = (py + step * gy) / (1.0 + step * norm)
    div = np.zeros_like(f)
    div[:-1, :] += px[:-1, :]
    div[1:, :] -= px[:-1, :]
    div[:, :-1] += py[:, :-1]
    div[:, 1:] -= py[:, :-1]
    return f - lam * div


def perimeter_map(llr: np.ndarray, lam: float, n_iter: int = 200) -> np.ndarray:
    """Global MAP of `max_S sum_{i in S} llr_i - lam * Per(S)`.

    Exact up to the ROF solver's convergence, via the level-set property in the
    module docstring. `llr` is signed evidence: positive means the pixel alone
    argues for being masked.
    """
    return rof(np.asarray(llr, dtype=np.float64), float(lam), n_iter) > 0.0


def perimeter_cost(mask: np.ndarray) -> int:
    """Length of the 4-connected boundary of `mask`, in pixel edges."""
    m = np.asarray(mask, bool)
    return int((m[:-1, :] != m[1:, :]).sum() + (m[:, :-1] != m[:, 1:]).sum())


# ==========================================================================
#  streak -- MAP under a curve prior
# ==========================================================================
@dataclass
class Streak:
    angle_deg: float
    offset: float
    score: float
    width: float
    extent: Tuple[int, int] = (0, 0)


def _pad_width(shape) -> int:
    """Padding that keeps every rotation of `shape` inside the frame."""
    return int(np.ceil(np.hypot(*shape) / 2 - min(shape) / 2)) + 1


def radon(field: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Line sums of `field` at each angle, `(n_angles, n_offsets)`.

    Rotate-and-sum rather than a dedicated transform: the evidence field is
    already smooth at the pixel scale, and bilinear rotation is accurate enough
    that the peak location is set by the streak, not by the interpolation.
    """
    f = np.nan_to_num(np.asarray(field, dtype=np.float64))
    g = np.pad(f, _pad_width(f.shape))
    return np.stack([ndi.rotate(g, a, reshape=False, order=1,
                                mode="constant", cval=0.0).sum(axis=0)
                     for a in angles], axis=0)


def curve_map(llr: np.ndarray, widths: Sequence[int] = (1, 2, 3, 5, 8),
              lam_n: float = 0.0, lam_w: float = 0.0, n_angles: int = 180,
              max_streaks: int = 8, suppress: int = 8,
              suppress_angle: int = 5) -> Tuple[np.ndarray, list]:
    """MAP under `Phi_curve = lambda_w * w + lambda_n`, searching over width.

    A line hypothesis' evidence is the SUM of `llr` along it -- which is why
    this detector must see the continuous field. The prior charges `lam_n` per
    accepted streak and `lam_w` per unit width, and NOTHING for length, so a
    long faint streak and a short bright one are judged on total evidence alone.

    Width is searched, not fixed. It is a free parameter of the prior, and
    getting it wrong is not a small error: a band one pixel wider than the
    streak adds a column of pure background, whose LLR is large and negative,
    and the hypothesis is rejected even though the streak is obvious. A
    width-`w` line sum is a box filter along the sinogram's offset axis, so the
    whole bank costs one Radon transform.

    Scores are standardized against the field's OWN background rather than
    against zero: an LLR field has a large negative mean (that is what the prior
    odds are), so an unstandardized line sum ranks long chords last regardless
    of what is on them. Subtracting `N * median` and dividing by
    `MAD * sqrt(N)` makes a null line `N(0, 1)`, which lets `lam_n` be read as a
    sigma threshold.

    Peaks are taken greedily with local suppression in both sinogram axes --
    the standard approximation to the multi-line MAP, exact when the streaks do
    not overlap in Radon space.
    """
    llr = np.asarray(llr, dtype=np.float64)
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sino = radon(llr, angles)
    support = radon(np.ones_like(llr), angles)
    finite = llr[np.isfinite(llr)]
    level = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - level))) or 1.0

    widths = [int(w) for w in widths]
    cube = np.stack([
        ((ndi.uniform_filter1d(sino, w, axis=1, mode="constant") * w
          - level * ndi.uniform_filter1d(support, w, axis=1, mode="constant") * w)
         / (scale * np.sqrt(np.maximum(
             ndi.uniform_filter1d(support, w, axis=1, mode="constant") * w, 1.0)))
         - lam_w * w)
        for w in widths], axis=0)

    out = np.zeros(llr.shape, bool)
    found = []
    work = cube.copy()
    for _ in range(max_streaks):
        wi, a, o = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[wi, a, o] <= lam_n:
            break
        w = widths[wi]
        band, extent = _line_band(llr.shape, angles[a], o, w, llr)
        found.append(Streak(float(angles[a]), float(o), float(work[wi, a, o]),
                            float(w), extent))
        out |= band
        alo, ahi = max(a - suppress_angle, 0), min(a + suppress_angle + 1, work.shape[1])
        olo, ohi = max(o - suppress, 0), min(o + suppress + 1, work.shape[2])
        work[:, alo:ahi, olo:ohi] = -np.inf
    return out, found


def _line_band(shape, angle_deg: float, offset: float, width: float,
               llr: Optional[np.ndarray] = None):
    """The thickened line at `(angle, offset)`, trimmed to its supported extent.

    Built by INVERTING the same rotation `radon` applies, rather than by
    re-deriving the line's equation in image coordinates. Deriving it
    independently means matching scipy's rotation sense, origin and padding
    convention by hand, and getting any of the three wrong puts the band
    somewhere plausible but wrong -- a silent failure, since the detector still
    reports a confident peak.

    `Phi_curve` charges width and count but NOT length, so the unconstrained MAP
    is the full chord across the frame -- correct under the prior, and badly
    over-masking in practice, because a real streak has ends. Given `llr`, the
    extent is solved exactly instead of assumed: the maximum-sum contiguous
    segment (Kadane) of the evidence profile along the line is the MAP endpoints
    under the same prior, since adding a pixel costs nothing but its own
    negative evidence.
    """
    pad = _pad_width(shape)
    padded = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    # Matches `uniform_filter1d`'s centring, so the band drawn here is exactly
    # the set of offsets the score was computed over.
    w = max(int(round(width)), 1)
    lo = max(int(round(offset - (w - 1) / 2.0)), 0)

    extent = (0, padded[0])
    if llr is not None:
        # NEAREST-neighbour rotation here, unlike the bilinear one in `radon`.
        # Detection wants smooth interpolation to localize the peak; the extent
        # must not dilute. Bilinear rotation of a 3-pixel line spreads it over
        # four columns and halves its amplitude, which on a marginal streak is
        # the difference between a positive and a negative evidence profile --
        # measured: mean profile +0.34 at order 0 against -0.24 at order 1.
        g = ndi.rotate(np.pad(np.nan_to_num(llr), pad), angle_deg, reshape=False,
                       order=0, mode="constant", cval=0.0)
        # The sinogram peak localizes the offset only to the nearest pixel, and
        # a one-pixel slip costs the whole streak (measured: extent 100 rows at
        # the right offset, 8 one pixel over). Refine by the segment's total
        # evidence -- not by its peak, which any single noise spike wins.
        best = None
        for s in range(max(lo - 1, 0), min(lo + 2, padded[1] - w + 1)):
            lo_s, hi_s, total = _max_segment(g[:, s:s + w].sum(axis=1))
            if best is None or total > best[0]:
                best = (total, s, (lo_s, hi_s))
        _, lo, extent = best

    hi = max(min(lo + w, padded[1]), 0)
    stripe = np.zeros(padded)
    stripe[extent[0]:extent[1], lo:hi] = 1.0
    back = ndi.rotate(stripe, -angle_deg, reshape=False, order=0,
                      mode="constant", cval=0.0)
    return back[pad:pad + shape[0], pad:pad + shape[1]] > 0.5, extent


def _max_segment(profile: np.ndarray) -> Tuple[int, int, float]:
    """`(lo, hi, sum)` of the maximum-sum contiguous segment of `profile`."""
    best = cur = -np.inf
    best_lo = best_hi = cur_lo = 0
    for i, v in enumerate(profile):
        if cur <= 0:
            cur, cur_lo = v, i
        else:
            cur += v
        if cur > best:
            best, best_lo, best_hi = cur, cur_lo, i + 1
    if not np.isfinite(best):
        return 0, int(profile.size), 0.0
    return best_lo, best_hi, float(best)


# ==========================================================================
#  blob -- matched-filter bank
# ==========================================================================
def blob_bank(t: np.ndarray, radii: Sequence[int] = (1, 2, 3, 5, 8),
              valid: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """`max_R l_R(i)` over a bank of disk filters, with the `sqrt(N_R)` whitening.

    Acts on the STANDARDIZED field `t`, not on the LLR: the matched filter for a
    constant-amplitude disk in white noise is the disk sum divided by
    `sqrt(N_R)`, which is a z-score for the disk and directly comparable across
    radii. Returns `(best score, best radius)` per pixel.
    """
    t = np.nan_to_num(np.asarray(t, dtype=np.float64))
    w = np.ones(t.shape) if valid is None else np.asarray(valid, float)
    best = np.full(t.shape, -np.inf)
    arg = np.zeros(t.shape, dtype=np.int32)
    for R in radii:
        k = _disk(R)
        s = ndi.convolve(t * w, k, mode="constant", cval=0.0)
        n = ndi.convolve(w, k, mode="constant", cval=0.0)
        z = s / np.sqrt(np.maximum(n, 1.0))
        upd = z > best
        best = np.where(upd, z, best)
        arg = np.where(upd, R, arg)
    return best, arg


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return ((x * x + y * y) <= r * r).astype(np.float64)


# ==========================================================================
#  the greedy baseline the pipeline currently uses
# ==========================================================================
def greedy_mask(t: np.ndarray, k: float, open_size: int = 2,
                close_size: int = 2) -> np.ndarray:
    """Threshold-then-regularize, the shape of `masking.Pipeline`.

    The comparison target for `perimeter_map`: identical evidence, but the shape
    prior is applied to the binary array after the decision instead of entering
    the decision. Morphology stands in for the pipeline's `mask`-kind
    regularizers (`regularization/pad.py`, `fill_holes.py`, `area_gate.py`).
    """
    m = np.nan_to_num(np.asarray(t, dtype=np.float64)) > k
    if open_size > 0:
        m = ndi.binary_opening(m, np.ones((open_size, open_size), bool))
    if close_size > 0:
        m = ndi.binary_closing(m, np.ones((close_size, close_size), bool))
    return m
