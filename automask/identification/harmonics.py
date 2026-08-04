"""
identification/harmonics.py -- the azimuthal harmonic cut, and its price.

The proposal: inside a ring, expand `log S(q, chi) = sum_m s_m(q) exp(i m chi)`
and project out `|m| <= m0`. The argument is that the three things living in a
ring separate by harmonic order --

    polarization      exactly m = +-2, removable analytically
    sample anisotropy low order in m
    a sharp shadow    broadband, out to m ~ 2 pi / delta_chi

-- so one interpretable knob `m0` replaces several opaque ones. Two parts of
that are checkable without any data at all, and this module makes them
checkable.

POLARIZATION IS m = +-2 IN THE LINEAR DOMAIN, NOT IN THE LOG. For a horizontally
polarized beam the correction is `P = 1 - p sin^2(2theta) cos^2(chi)`, and
`cos^2 chi = (1 + cos 2chi)/2` is exactly two harmonics. But the cut is proposed
on `log S`, and `log(a - b cos 2chi)` is not band-limited: expanding it produces
every even harmonic with amplitude falling like `(b/a)^k / k`. `log_pol_leakage`
computes how much power that puts above `m = 2` at this experiment's actual
`2theta` range. Whether the leakage matters is an empirical question about its
size relative to a shadow, which is why it is measured rather than argued.

PARTIAL AZIMUTHAL COVERAGE CAPS m0. The beam sits near a detector corner, so a
ring covers an arc, not a circle. Harmonics are only orthogonal over the full
circle; on an arc of width `W` the design matrix for orders up to `m0` becomes
ill-conditioned once the orders stop being resolvable, roughly `m0 W < pi`, and
a projection computed through an ill-conditioned design does not remove the
harmonics, it amplifies the noise. `fit_harmonics` therefore always returns the
condition number of the design it used, and `max_usable_order` reports where it
crosses a tolerance. This is a hard geometric limit on the method, independent
of statistics.

A SHARP SHADOW IS BROADBAND, AND THAT CUTS BOTH WAYS. `tophat_spectrum` is the
analytic harmonic content of an angular top-hat of width `delta_chi`: power
`~ sinc^2(m delta_chi / 2)`, with the first zero at `m = 2 pi / delta_chi`. So a
narrow shadow survives the cut, as claimed. The corollary the notes state --
blindness to shadows wider than `~2 pi / m0` -- is quantified by
`surviving_fraction`, which is the curve that should set `m0`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Condition number above which a harmonic projection is considered unusable.
#: Chosen as the point where a projection amplifies input noise by more than an
#: order of magnitude, which is well before the numerics break.
COND_TOL = 10.0


def design(chi: np.ndarray, m_max: int) -> np.ndarray:
    """Real Fourier design `[1, cos chi, sin chi, ..., cos m chi, sin m chi]`."""
    chi = np.asarray(chi, dtype=np.float64).ravel()
    cols = [np.ones_like(chi)]
    for m in range(1, m_max + 1):
        cols.append(np.cos(m * chi))
        cols.append(np.sin(m * chi))
    return np.stack(cols, axis=1)


def fit_harmonics(chi: np.ndarray, y: np.ndarray, m_max: int,
                  valid: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares harmonic coefficients of `y(chi)` up to `m_max`.

    Returns `(coeffs, residual, cond)`. `residual` is full-length with NaN
    outside `valid`, so it can be handed straight back to a per-pixel detector.
    `cond` is the condition number of the design over the sampled `chi` -- the
    number that decides whether the fit means anything on a partial arc.
    """
    chi = np.asarray(chi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.ones(chi.shape, bool) if valid is None else np.asarray(valid, bool)
    use = valid & np.isfinite(y) & np.isfinite(chi)
    X = design(chi[use], m_max)
    cond = float(np.linalg.cond(X)) if X.size and X.shape[0] > X.shape[1] else np.inf
    if not np.isfinite(cond):
        return np.full(2 * m_max + 1, np.nan), np.full(chi.shape, np.nan), cond
    coeffs, *_ = np.linalg.lstsq(X, y[use], rcond=None)
    resid = np.full(chi.shape, np.nan)
    resid[use] = y[use] - X @ coeffs
    return coeffs, resid, cond


def power_spectrum(coeffs: np.ndarray) -> np.ndarray:
    """Per-order power `a_m^2 + b_m^2`, with the `m = 0` term first."""
    c = np.asarray(coeffs, dtype=np.float64)
    m_max = (c.size - 1) // 2
    out = np.empty(m_max + 1)
    out[0] = c[0] ** 2
    for m in range(1, m_max + 1):
        out[m] = c[2 * m - 1] ** 2 + c[2 * m] ** 2
    return out


def angular_coverage(chi: np.ndarray, gap_factor: float = 20.0
                     ) -> Tuple[float, float]:
    """`(covered arc, largest gap)` in radians, for angles on the circle.

    `max - min` is the wrong measure and it is wrong in the flattering
    direction: a ring covering two opposite 60-degree arcs has `max - min` close
    to `2 pi` while covering a third of the circle. So is `2 pi` minus the
    LARGEST gap, which credits every arc after the first with the space between
    them. Coverage here is `2 pi` minus the total length of all gaps wider than
    `gap_factor` times the typical spacing between neighbouring samples -- the
    measure of the circle the ring actually visits, however many pieces it comes
    in.
    """
    c = np.sort(np.mod(np.asarray(chi, dtype=np.float64).ravel(), 2 * np.pi))
    if c.size < 2:
        return 0.0, 2 * np.pi
    gaps = np.append(np.diff(c), c[0] + 2 * np.pi - c[-1])
    typical = float(np.median(gaps))
    holes = gaps[gaps > gap_factor * max(typical, 1e-12)]
    return 2 * np.pi - float(holes.sum()), float(gaps.max())


def arc_phase(chi: np.ndarray) -> np.ndarray:
    """Map angles onto `[0, 2pi)` by stretching the arc they actually cover.

    The harmonics of the full circle are not orthogonal on an arc, which is what
    destroys the conditioning. Rescaling the covered arc to a full period
    restores it, at the cost of reinterpreting the order: on an arc of width `W`,
    order `m` now resolves angular features of size `W/m` rather than `2 pi/m`.
    That reinterpretation is the honest form of the harmonic cut on this
    geometry -- see `exp_h3_coverage_limit`.
    """
    c = np.mod(np.asarray(chi, dtype=np.float64), 2 * np.pi)
    flat = np.sort(c.ravel())
    gaps = np.diff(flat)
    wrap = flat[0] + 2 * np.pi - flat[-1]
    start = flat[0] if wrap >= gaps.max() else flat[int(np.argmax(gaps)) + 1]
    rel = np.mod(c - start, 2 * np.pi)
    span = rel.max()
    return rel * (2 * np.pi / span) if span > 0 else rel


def max_usable_order(chi: np.ndarray, valid: Optional[np.ndarray] = None,
                     m_limit: int = 40, tol: float = COND_TOL,
                     on_arc: bool = False) -> int:
    """Largest `m0` whose design over the sampled `chi` stays better conditioned
    than `tol`. Returns 0 when even the first harmonic is unresolvable.

    With `on_arc`, the angles are stretched by `arc_phase` first, which is what
    makes the fit computable at all on partial coverage.
    """
    chi = np.asarray(chi, dtype=np.float64)
    valid = np.ones(chi.shape, bool) if valid is None else np.asarray(valid, bool)
    c = chi[valid & np.isfinite(chi)]
    if c.size < 4:
        return 0
    if on_arc:
        c = arc_phase(c)
    best = 0
    for m in range(1, m_limit + 1):
        X = design(c, m)
        if X.shape[0] <= X.shape[1] or np.linalg.cond(X) > tol:
            break
        best = m
    return best


# ==========================================================================
#  analytic checks -- no data required
# ==========================================================================
def polarization(two_theta: np.ndarray, chi: np.ndarray, p: float = 1.0):
    """Thomson polarization factor for a beam polarized along `chi = 0`.

    `P = 1 - p sin^2(2theta) cos^2(chi)`, the form pyFAI's `polarization()`
    implements and the one `automask.azimuthal` validated against psana's own
    `azav__azav_pol` array to 1e-4.
    """
    return 1.0 - p * np.sin(np.asarray(two_theta)) ** 2 * np.cos(np.asarray(chi)) ** 2


def log_pol_leakage(two_theta: float, p: float = 1.0, m_max: int = 12,
                    n_chi: int = 4096) -> dict:
    """Harmonic content of `log P` at one scattering angle, over the full circle.

    In the linear domain `P` is exactly `m = 0` plus `m = 2`. The quantity
    reported here is `leak`: the fraction of `log P`'s non-constant power that
    sits ABOVE `m = 2`, i.e. the part an analytic `m = +-2` removal on the log
    field would leave behind.
    """
    chi = np.linspace(0.0, 2.0 * np.pi, n_chi, endpoint=False)
    P = polarization(np.full_like(chi, two_theta), chi, p)
    lin = power_spectrum(fit_harmonics(chi, P, m_max)[0])
    log = power_spectrum(fit_harmonics(chi, np.log(P), m_max)[0])
    ac_lin, ac_log = lin[1:], log[1:]
    return {"two_theta_deg": float(np.degrees(two_theta)),
            "linear_above_m2": float(ac_lin[2:].sum() / max(ac_lin.sum(), 1e-300)),
            "log_above_m2": float(ac_log[2:].sum() / max(ac_log.sum(), 1e-300)),
            "log_m4_over_m2": float(np.sqrt(log[4] / max(log[2], 1e-300))),
            "log_depth": float(np.log(P.max() / P.min()))}


def tophat_spectrum(delta_chi: float, m_max: int = 64) -> np.ndarray:
    """Harmonic power of an angular top-hat of width `delta_chi`, orders 0..m_max.

    The Fourier coefficient of a width-`w` top-hat on the circle is
    `(w/2pi) sinc(m w / 2)`, so the power falls as `sinc^2` with its first zero
    at `m = 2 pi / w` -- the "broadband to `m ~ 2 pi / delta_chi`" claim, made
    numerical.
    """
    m = np.arange(m_max + 1, dtype=np.float64)
    w = float(delta_chi)
    return (w / (2 * np.pi)) ** 2 * np.sinc(m * w / (2 * np.pi)) ** 2


def surviving_fraction(delta_chi: float, m0: int, m_max: int = 256) -> float:
    """Fraction of a width-`delta_chi` shadow's power left after cutting `|m| <= m0`.

    The cost side of the harmonic cut: this is the detectability a shadow of
    that angular size retains, and it collapses once `m0 > 2 pi / delta_chi`.
    """
    P = tophat_spectrum(delta_chi, m_max)
    ac = P[1:]
    total = ac.sum()
    return float(ac[m0:].sum() / total) if total > 0 else float("nan")
