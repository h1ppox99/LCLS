"""
identification/twoway.py -- what the condition contrast can and cannot recover.

Given flux-normalised per-condition per-pixel means

    zbar_i^(c) = tau_i * A_i * S(q_i; theta_c)  +  J_i        (+ noise)

this module implements the three things the notes claim about that expression,
in the order the composition forces.

WHAT IS IDENTIFIABLE. Write the model with `J = 0` in logs and it is a two-way
fixed-effects layout: `y_ic = alpha_i + gamma_{q_i, c}`. For any ring `q` and any
constant `k`, moving `alpha_i -> alpha_i + k` for every pixel in that ring and
`gamma_{q,c} -> gamma_{q,c} - k` for every `c` leaves every prediction
unchanged. The design is therefore rank-deficient by exactly one per ring, and
NO estimator recovers the ring-constant part of `alpha`. `ring_split` makes that
concrete: it decomposes any per-pixel field into the part the data can speak
about (within-ring) and the part it cannot (ring-constant). A shadow that covers
a whole ring uniformly is invisible to this model -- not hard to see, invisible.

HOW MUCH IDENTIFYING POWER A RUN HAS. `tau` separates from `S` only through the
CONTRAST between conditions, so the precision of any recovered `tau` scales as
the inverse of how much the conditions move the signal. `identifying_power`
measures `Var_c(gamma_{q,c})` per ring and reports it against its own sampling
floor, which is the honest form: `Var_c` estimated from G group means sits above
its expectation half the time even when the signal does not move at all, so the
raw variance is not interpretable and the excess over the floor is.

THE TWO STAGES, AND WHY THE ORDER IS NOT A CHOICE. `J_i` enters `zbar` with no
condition dependence at all, so DIFFERENCING two conditions cancels it exactly --
not approximately, not to first order, exactly, because it is the same number in
both terms. That makes the shadow recoverable without knowing anything about the
parasitic scattering. The reverse is false: any residual built to expose `J`
needs a prediction of `tau_i A_i S`, so it needs `tau` already. Hence stage 1
then stage 2, and `exp_m4_stage_order` in the study measures what running them
the other way costs.

Every ring reduction here is a MEDIAN, not a mean. The pixels a ring is reduced
over include the very defects the estimator is looking for, so a mean lets a
shadow set the reference it is being compared against; this is the same failure
`unsupervised.azimuthal.ring_reference` documents for the noise scale, in the
first moment instead of the second. `exp_a4_robust_reduction` measures the
difference rather than asserting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: Minimum pixels a ring needs before any reduction over it is reported.
N_MIN_RING = 30

#: Ring reductions available to every estimator here. `median` is the default
#: for the reason in the module docstring; the others exist so the choice can be
#: measured (`exp_a4_robust_reduction`) instead of asserted.
REDUCERS = ("median", "mean", "trimmed")


def reduce_rings(values: np.ndarray, ring_idx: np.ndarray, valid: np.ndarray,
                 n_rings: int, how: str = "median",
                 trim: float = 0.25) -> np.ndarray:
    """Per-ring scalar summary of a per-pixel field. Empty rings give NaN."""
    out = np.full(n_rings, np.nan)
    r = ring_idx[valid]
    v = values[valid]
    ok = np.isfinite(v)
    r, v = r[ok], v[ok]
    if r.size == 0:
        return out
    order = np.argsort(r, kind="stable")
    r, v = r[order], v[order]
    bounds = np.searchsorted(r, np.arange(n_rings + 1))
    for k in range(n_rings):
        seg = v[bounds[k]:bounds[k + 1]]
        if seg.size < N_MIN_RING:
            continue
        if how == "mean":
            out[k] = seg.mean()
        elif how == "median":
            out[k] = np.median(seg)
        elif how == "trimmed":
            lo, hi = np.quantile(seg, [trim, 1.0 - trim])
            keep = seg[(seg >= lo) & (seg <= hi)]
            out[k] = keep.mean() if keep.size else np.nan
        else:
            raise ValueError(f"unknown reducer {how!r}; known: {list(REDUCERS)}")
    return out


def ring_trend(values: np.ndarray, x: np.ndarray, ring_idx: np.ndarray,
               valid: np.ndarray, n_rings: int, n_sigma: float = 3.0,
               n_iter: int = 2) -> np.ndarray:
    """Per-pixel robust linear fit of `values` against `x` WITHIN each ring.

    A ring is a bin, not a point: `q` varies across it, and any quantity with a
    radial slope therefore varies across it too. Dividing by a single per-ring
    constant charges that variation to the pixel, which is precisely the failure
    mode a transmission estimate cannot afford -- the leakage is smooth, large,
    and looks exactly like a shadow with soft edges. Fitting `a + b x` inside
    the ring removes it to first order; the fit is refit after rejecting
    `n_sigma` MAD outliers so the artifacts the estimator is hunting do not set
    their own baseline.

    Returns a full-size array of fitted values, NaN outside `valid`.
    """
    out = np.full(values.shape, np.nan)
    for k in range(n_rings):
        sel = valid & (ring_idx == k) & np.isfinite(values) & np.isfinite(x)
        if sel.sum() < N_MIN_RING:
            continue
        xv, yv = x[sel], values[sel]
        keep = np.ones(xv.size, bool)
        coef = (np.median(yv), 0.0)
        for _ in range(n_iter + 1):
            if keep.sum() < 3:
                break
            A = np.stack([np.ones(keep.sum()), xv[keep]], axis=1)
            coef = np.linalg.lstsq(A, yv[keep], rcond=None)[0]
            resid = yv - (coef[0] + coef[1] * xv)
            mad = np.median(np.abs(resid - np.median(resid)))
            if mad <= 0:
                break
            keep = np.abs(resid - np.median(resid)) <= n_sigma * 1.4826 * mad
        out[sel] = coef[0] + coef[1] * xv
    return out


def ring_split(field: np.ndarray, ring_idx: np.ndarray, valid: np.ndarray,
               n_rings: int, how: str = "median") -> Tuple[np.ndarray, np.ndarray]:
    """Split a per-pixel field into its (ring-constant, within-ring) parts.

    The first is the component the two-way model cannot identify; the second is
    everything an azimuthal method could ever see. Returned as full-size arrays
    so they can be compared, plotted and scored against a reference mask.
    """
    per_ring = reduce_rings(field, ring_idx, valid, n_rings, how)
    ring_part = np.where(valid, per_ring[ring_idx], np.nan)
    return ring_part, np.where(valid, field - ring_part, np.nan)


def identifiable_fraction(field: np.ndarray, ring_idx: np.ndarray,
                          valid: np.ndarray, n_rings: int) -> float:
    """Fraction of a field's energy that survives the ring projection.

    1.0 means the structure is purely azimuthal and fully visible to the model;
    0.0 means it is a radial profile and completely confounded with `S(q)`. Run
    it on a reference mask's signature to find out, before building anything,
    how much of the real artifact this whole approach is even entitled to find.
    """
    ring_part, within = ring_split(field, ring_idx, valid, n_rings, how="mean")
    ok = valid & np.isfinite(within) & np.isfinite(ring_part)
    total = float(np.nansum(np.where(ok, field - np.nanmean(field[ok]), 0.0) ** 2))
    if total <= 0:
        return float("nan")
    return float(np.nansum(np.where(ok, within, 0.0) ** 2) / total)


def design_rank_deficiency(n_pixels_per_ring: Sequence[int],
                           n_conditions: int) -> dict:
    """Predicted rank of the two-way design `y_ic = alpha_i + gamma_{q,c}`.

    Parameters are `sum(n_pixels_per_ring)` pixel effects plus
    `n_rings * n_conditions` ring-by-condition effects. One shift per ring is
    unobservable, plus the usual global intercept which those shifts already
    contain -- so the deficiency is exactly `n_rings`. Returned as numbers so
    the study can check them against an SVD of the real design matrix rather
    than trusting the algebra.
    """
    counts = np.asarray(list(n_pixels_per_ring), dtype=int)
    n_rings = counts.size
    n_params = int(counts.sum()) + n_rings * n_conditions
    return {"n_params": n_params, "n_rings": n_rings,
            "predicted_rank": n_params - n_rings,
            "predicted_deficiency": n_rings}


# ==========================================================================
#  identifying power
# ==========================================================================
@dataclass
class IdentifyingPower:
    """Per-ring measurement of how much the conditions move the signal.

    `var_gamma` is `Var_c(log S(q; theta_c))` estimated from the ring's own
    condition means, `floor` is what that variance would read under "the signal
    does not move at all", and `excess` is the difference clipped at zero. The
    headline is `snr = excess / floor`: below 1 the run carries no usable
    contrast and stage 1 cannot run, whatever the estimator.
    """
    q: np.ndarray
    gamma: np.ndarray                # (n_rings, G)
    var_gamma: np.ndarray
    floor: np.ndarray
    excess: np.ndarray
    snr: np.ndarray

    def summary(self) -> dict:
        ok = np.isfinite(self.snr)
        return {"n_rings": int(ok.sum()),
                "median_snr": float(np.nanmedian(self.snr[ok])) if ok.any() else float("nan"),
                "median_excess": float(np.nanmedian(self.excess[ok])) if ok.any() else float("nan"),
                "frac_rings_snr_gt1": float(np.mean(self.snr[ok] > 1.0)) if ok.any() else float("nan")}


def identifying_power(zbar: np.ndarray, A: np.ndarray, ring_idx: np.ndarray,
                      valid: np.ndarray, n_rings: int,
                      sem2: Optional[np.ndarray] = None,
                      q: Optional[np.ndarray] = None,
                      how: str = "median") -> IdentifyingPower:
    """`Var_c(gamma_{q,c})` per ring, against its own sampling floor.

    `zbar` is `(G, ...)` flux-normalised condition means, `A` the known
    solid-angle x polarization factor, `sem2` the squared standard error of each
    condition mean (`GroupMoments.sem2()`). Rings whose signal is not strictly
    positive are dropped rather than clipped: `gamma` is a log.

    `gamma` is estimated from PER-PIXEL RATIOS to the pixel's own across-
    condition mean, then reduced over the ring -- not from the ring level of
    each condition separately. Reducing first and dividing after leaves the
    ring's radial spread inside the reduction, and since that spread is orders
    of magnitude larger than the counting noise, the sampling floor becomes
    unusable (measured: the ring-median route reads SNR ~ 50 on a field built
    with zero condition contrast). Ratios cancel `tau_i A_i S(q_i)` pixel-wise
    first, so what the ring reduces is already a pure condition effect.

    One bias worth naming: with `J != 0` the ratio is
    `(tau A S_c + J)/(tau A S_bar + J)`, which is pulled toward 1. Parasitic
    scattering therefore makes this measurement CONSERVATIVE -- it understates
    the identifying power rather than inventing it.
    """
    G = zbar.shape[0]
    z = np.asarray(zbar, dtype=np.float64)
    ok = np.isfinite(z)
    n_ok = ok.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_mean = np.where(n_ok > 0,
                              np.where(ok, z, 0.0).sum(axis=0) / np.maximum(n_ok, 1),
                              np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(pixel_mean) > 0, z / pixel_mean, np.nan)
    g = np.stack([reduce_rings(ratio[c], ring_idx, valid, n_rings, how)
                  for c in range(G)], axis=1)
    good = np.all(np.isfinite(g) & (g > 0), axis=1)
    gamma = np.where(good[:, None], np.log(np.where(g > 0, g, np.nan)), np.nan)
    var_gamma = np.nanvar(gamma, axis=1, ddof=1)

    if sem2 is None:
        floor = np.full(n_rings, np.nan)
    else:
        # Each ratio carries sem2/pixel_mean^2; a median over N of them has
        # pi/2 times the variance of a mean, and gamma is a log of something
        # centred on 1, so no further scaling is needed.
        counts = np.bincount(ring_idx[valid], minlength=n_rings).astype(float)
        rel = np.stack([reduce_rings(sem2[c] / np.maximum(pixel_mean ** 2, 1e-300),
                                     ring_idx, valid, n_rings, how="mean")
                        for c in range(G)], axis=1)
        floor = np.where(good, (np.pi / 2) * rel.mean(axis=1)
                         / np.maximum(counts, 1), np.nan)

    excess = np.maximum(var_gamma - floor, 0.0)
    return IdentifyingPower(
        q=(reduce_rings(q, ring_idx, valid, n_rings, "mean")
           if q is not None else np.arange(n_rings, dtype=float)),
        gamma=gamma, var_gamma=var_gamma, floor=floor, excess=excess,
        snr=excess / np.where(floor > 0, floor, np.nan))


# ==========================================================================
#  stage 1 -- shadows
# ==========================================================================
@dataclass
class Stage1:
    """Recovered relative transmission and its evidence field.

    `tau_rel` estimates `tau_i / tau_bar_{q_i}` -- the ring-constant part is not
    recoverable and is not claimed. `t` is the signed standardized deviation
    from the ring's own transmission, positive for a pixel that transmits LESS
    than its ring, which is the polarity a shadow has. `contrast` is the ring's
    `|S_c - S_c'|`, i.e. how much identifying power this particular pair of
    conditions supplied; where it is small, `tau_rel` is noise and `t` says so
    through its own error bar.
    """
    tau_rel: np.ndarray
    sigma: np.ndarray
    t: np.ndarray
    contrast: np.ndarray
    pair: Tuple[int, int]


def stage1_transmission(zbar: np.ndarray, A: np.ndarray, ring_idx: np.ndarray,
                        valid: np.ndarray, n_rings: int, pair: Tuple[int, int],
                        sem2: Optional[np.ndarray] = None,
                        how: str = "median", detrend: str = "linear",
                        q: Optional[np.ndarray] = None) -> Stage1:
    """Relative transmission from the contrast between two conditions.

    `delta_i = zbar_i^(c) - zbar_i^(c')` kills `J_i` identically, leaving
    `tau_i A_i [S_c - S_c']`. Dividing by the known `A_i` and by the ring's own
    reduction of the same quantity leaves `tau_i / tau_bar_q`.

    `detrend="linear"` fits that reduction as `a + b q` inside each ring rather
    than as a constant. Measured on the synthetic bench, the constant form is
    not noise-limited at all: its error is 6-10x the propagated counting noise
    everywhere except at the extremum of `S_c - S_c'`, where the radial slope
    happens to vanish. Pass `detrend="ring"` to reproduce the constant form --
    `exp_m2_identifying_power` reports both, because the difference between them
    is what decides whether identifying power is really the binding constraint.
    """
    c, cp = pair
    delta = (zbar[c] - zbar[cp]) / A
    ring = reduce_rings(delta, ring_idx, valid, n_rings, how)
    if detrend == "linear":
        if q is None:
            raise ValueError("detrend='linear' needs the per-pixel q map")
        denom = ring_trend(delta, q, ring_idx, valid, n_rings)
    elif detrend == "ring":
        denom = np.where(valid, ring[ring_idx], np.nan)
    else:
        raise ValueError(f"unknown detrend {detrend!r}; use 'linear' or 'ring'")
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_rel = np.where(np.abs(denom) > 0, delta / denom, np.nan)

    if sem2 is None:
        sigma = np.full_like(tau_rel, np.nan)
    else:
        var_delta = (sem2[c] + sem2[cp]) / A ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.sqrt(var_delta) / np.abs(denom)
    # `tau_rel` is invariant to the sign of the ring contrast -- numerator and
    # denominator flip together -- so the deviation needs no sign convention.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.isfinite(sigma) & (sigma > 0),
                     (1.0 - tau_rel) / sigma, np.nan)
    return Stage1(tau_rel=tau_rel, sigma=sigma, t=t,
                  contrast=ring, pair=(c, cp))


# ==========================================================================
#  stage 2 -- streaks
# ==========================================================================
@dataclass
class Stage2:
    """Recovered parasitic scattering, in flux-normalised intensity units.

    `J` is identified only up to its ring median, for the same reason `tau` is:
    a parasitic term constant around a ring is indistinguishable from a slightly
    brighter sample. `t` standardizes it by the measured sampling error.
    """
    J: np.ndarray
    sigma: np.ndarray
    t: np.ndarray
    condition: int


def stage2_parasitic(zbar: np.ndarray, A: np.ndarray, ring_idx: np.ndarray,
                     valid: np.ndarray, n_rings: int, tau_rel: np.ndarray,
                     condition: int = 0, sem2: Optional[np.ndarray] = None,
                     how: str = "median", detrend: str = "linear",
                     q: Optional[np.ndarray] = None) -> Stage2:
    """Residual of one condition against the recovered transmission.

    `S` is not known, so what is subtracted is the ring reduction of
    `zbar / (tau_rel * A)` -- the best available estimate of `tau_bar_q S_c(q)`.
    Pixels whose recovered transmission is near zero are dropped instead of
    dividing through: a fully opaque pixel carries no information about what is
    added on top of nothing.

    `detrend` behaves as in `stage1_transmission`, and matters more here: the
    residual is taken in intensity units, so an unmodelled radial slope inside a
    ring appears directly as a false `J`.
    """
    z = zbar[condition]
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(np.abs(tau_rel) > 1e-3, z / (tau_rel * A), np.nan)
    if detrend == "linear":
        if q is None:
            raise ValueError("detrend='linear' needs the per-pixel q map")
        S_pix = ring_trend(implied, q, ring_idx, valid, n_rings)
    else:
        S_pix = np.where(valid, reduce_rings(implied, ring_idx, valid, n_rings,
                                             how)[ring_idx], np.nan)
    pred = np.where(valid, tau_rel * A * S_pix, np.nan)
    J = np.where(valid, z - pred, np.nan)

    if sem2 is None:
        sigma = np.full_like(J, np.nan)
    else:
        sigma = np.sqrt(sem2[condition])
    t = np.where(np.isfinite(sigma) & (sigma > 0), J / sigma, np.nan)
    return Stage2(J=J, sigma=sigma, t=t, condition=condition)


def llr(t: np.ndarray, prior_odds: float = 1e-3,
        amplitude: Optional[float] = None) -> np.ndarray:
    """Per-pixel log-likelihood ratio for "this pixel is bad", from a z-score.

    Under the null `t ~ N(0,1)`. Two alternatives, and the choice is not
    cosmetic -- it decides whether a shape prior can do anything at all:

    * `amplitude=None` -- the deviation is real but of unknown size and sign, so
      the LLR is `t^2/2 + log(prior odds)`. Two-sided, and appropriate when
      hot and dead pixels are both in scope.
    * `amplitude=a` -- the deviation is a known `a` sigma in a known direction,
      giving the matched-filter form `a t - a^2/2 + log(prior odds)`.

    The two coincide at `t = a` (the quadratic is the linear form's tangent
    there), so the difference is not about how strongly a nominal artifact
    scores. It is about everything else. The quadratic is unsigned, so a pixel
    that reads `a` sigma LOW scores exactly as a pixel that reads `a` sigma
    high; and its null is `chi^2_1`, which is heavy-tailed, so isolated noise
    survives into any region sum. Measured on the study's bench at a matched
    pointwise threshold, the two-sided form keeps 96% of the isolated noise
    spikes where the matched form keeps 24%, for the same recovered blob.

    Use the two-sided form when hot and dead pixels are both in scope and the
    decision is per pixel; use the matched form for anything that sums evidence
    over a region, which is every prior in `identification.priors`.
    """
    t = np.asarray(t, dtype=np.float64)
    if amplitude is None:
        return 0.5 * t ** 2 + np.log(prior_odds)
    a = float(amplitude)
    return a * t - 0.5 * a ** 2 + np.log(prior_odds)
