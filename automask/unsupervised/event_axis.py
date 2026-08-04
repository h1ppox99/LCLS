"""
unsupervised/event_axis.py -- tier 3: a good pixel is stationary across events.

Tiers 0-2 all collapse the event axis before they look at anything: they see a
sum image, a mean, a std. This one keeps it. The hypothesis is that a working
pixel, watching a beam whose intensity wanders, measures ONE distribution whose
scale follows that wander -- so if the run is cut into K blocks of shots, the K
block means of a good pixel differ only by counting noise, once the beam's own
drift is divided out. A pixel that flickers, latches, or changes gain mid-run
fails that, and fails it in a way no time-averaged image can show: it can have a
perfectly ordinary mean and a perfectly ordinary variance.

THE STATISTIC. From the fold moments (`unsupervised.folds`), per pixel:

    g_k        = median over live pixels of m_k / mbar     global gain of fold k
    chi2       = sum_k (m_k/g_k - wbar)^2 / (se_k/g_k)^2   dof K-1
    z          = (chi2 - (K-1)) / sqrt(2(K-1))

Dividing by `g_k` is not cosmetic. The beam intensity drifts over a run by far
more than counting noise, so without it EVERY pixel rejects stationarity and the
statistic measures the machine, not the detector. `g_k` is a single number per
fold, estimated over ~1M pixels, so removing it costs essentially no power.

THE METRIC on a mask is a LEAK: of the pixels the mask leaves in service, what
fraction are anomalous at p < 1e-3? A perfect mask leaks 1e-3 (the false-positive
rate of the test itself); a mask that missed a flickering ASIC leaks far more.
As everywhere in this package the raw leak rewards masking more, so it is paired
with a size-matched random control (`event_gain`).

TWO HONEST LIMITATIONS.
  * BLIND TO CONSTANT PIXELS. A pixel stuck at a fixed value is perfectly
    stationary and scores perfectly here. That is not a bug to patch -- it is the
    hypothesis's edge -- and it is covered by the variance detector and by tier
    2, which see a dead pixel as a dark sector.
  * SHARED DATA WITH THE DETECTORS. This reads the same shots the `variance`
    stat does, so agreement between them is weaker evidence than agreement with
    tier 2's azimuthal argument, which rests on the scattering geometry instead.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from automask.unsupervised.base import MetricSpec, register_metric

P_ANOM = 1e-3       # per-pixel significance for "not stationary"


def anomaly_field(sample, fm):
    """Per-pixel stationarity chi2 across folds, in assembled space.

    Returns `(chi2, dof, testable)`; `testable` excludes pixels whose fold
    standard errors are all zero (constant pixels -- see the module docstring).
    """
    from automask.geometry import panel_to_asm

    mean, se = fm.fold_means()                     # (K, 2, 512, 1024) each
    k = mean.shape[0]
    live = mean.mean(axis=0) > 0
    # One scalar per fold: how much brighter the whole detector was in that fold.
    ref = mean.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.array([np.median((mean[i][live] / ref[live])) for i in range(k)])
    gain = np.where(np.isfinite(gain) & (gain > 0), gain, 1.0)

    m = mean / gain[:, None, None, None]
    s = se / gain[:, None, None, None]
    w = np.where(s > 0, 1.0 / np.maximum(s, 1e-12) ** 2, 0.0)
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = np.where(wsum > 0, (w * m).sum(axis=0) / np.maximum(wsum, 1e-300), 0.0)
    chi2 = (w * (m - mbar) ** 2).sum(axis=0)
    n_ok = (s > 0).sum(axis=0)
    testable = n_ok >= 3
    dof = np.maximum(n_ok - 1, 1)
    return (panel_to_asm(chi2, fm.run), panel_to_asm(dof, fm.run),
            panel_to_asm(testable, fm.run) & (sample.sumimg != 0))


def _flagged(ctx):
    """Boolean field of event-axis-anomalous pixels, plus the testable domain."""
    def build():
        fm = ctx.folds()
        chi2, dof, testable = anomaly_field(ctx.sample, fm)
        p = stats.chi2.sf(chi2, dof)
        return (testable & (p < P_ANOM)), testable
    return ctx.cached("event_anom", build)


def _leak(mask, flagged, testable) -> float:
    keep = testable & ~np.asarray(mask, bool)
    n = int(keep.sum())
    if n == 0:
        raise ValueError("mask leaves no testable pixel in service")
    return float((flagged & keep).sum()) / n


def event_leak(cand, ctx) -> float:
    """Fraction of the pixels left in service that reject stationarity."""
    flagged, testable = _flagged(ctx)
    return _leak(cand.mask(ctx), flagged, testable)


def event_gain(cand, ctx) -> float:
    """Leak of a size-matched random mask minus the candidate's.

    Positive means the mask removed pixels that were genuinely non-stationary,
    rather than merely removing pixels."""
    flagged, testable = _flagged(ctx)
    fr = ctx.azimuthal_frame()
    m = cand.mask(ctx)
    ctrl = fr.random_control(m, ctx.rng("event"))
    return _leak(ctrl, flagged, testable) - _leak(m, flagged, testable)


register_metric(MetricSpec(
    name="event_leak", compute=event_leak, higher_is_better=False, tier=3,
    doc="fraction of unmasked pixels failing a cross-fold stationarity chi2"))
register_metric(MetricSpec(
    name="event_gain", compute=event_gain, higher_is_better=True, tier=3,
    doc="stationarity leak removed, versus a size-matched random mask"))
