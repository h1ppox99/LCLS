"""
Tests for the label-free metric package. Same style as test_components.py --
plain asserts, tiny synthetic arrays, no frozen dataset and no psana:

    python -m automask.tests.test_unsupervised

These are not smoke tests. Each unsupervised metric asserts a statistical claim
("this is ~0 when the hypothesis holds", "this rejects at the stated rate"), and
a metric whose null is miscalibrated reports defects that are not there. So the
important cases here build data where the truth is known BY CONSTRUCTION -- pure
noise, or noise plus one injected anomaly -- and check the statistic against it.
"""
from __future__ import annotations

import numpy as np

from automask.unsupervised.azimuthal import cell_moments, excess_scatter
from automask.unsupervised.base import Candidate, iou
from automask.unsupervised.event_axis import P_ANOM
from automask.unsupervised.stability import jitter_params


# -- base ------------------------------------------------------------------
def test_iou_conventions():
    a = np.zeros((8, 8), bool); a[:4] = True
    b = np.zeros((8, 8), bool); b[2:6] = True
    assert abs(iou(a, b) - 2 / 6) < 1e-12
    assert iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0
    assert iou(a, a) == 1.0


def test_candidate_caches_per_run():
    calls = []

    def make(sample):
        calls.append(sample.run)
        return np.zeros((4, 4), bool)

    class S:                       # minimal stand-in for a Sample
        run = 475

    class Ctx:
        run = 475
        sample = S()

    c = Candidate("x", make)
    c.mask(Ctx()); c.mask(Ctx())
    assert calls == [475], "the mask must be built once per run, not per metric"


# -- tier 1: the knob jitter ----------------------------------------------
def test_jitter_respects_types_and_fixed_knobs():
    from automask.stats.asic_polish import AsicPolishParams

    rng = np.random.default_rng(0)
    p = AsicPolishParams(asic=256, n_iter=3, k=15.0, mode="high")
    out = [jitter_params(p, 0.2, rng) for _ in range(50)]
    assert all(o.asic == 256 for o in out), "ASIC size is hardware, never jittered"
    assert all(o.mode == "high" for o in out), "strings are left alone"
    assert all(isinstance(o.n_iter, int) and o.n_iter >= 1 for o in out)
    assert len({o.k for o in out}) > 40, "float knobs must actually move"
    # lognormal(0, eps) is centred on 1 in the log, so the median is the nominal
    assert abs(np.median([o.k for o in out]) / 15.0 - 1.0) < 0.15


def test_jitter_is_reproducible_from_the_seed():
    from automask.stats.variance import VarianceParams
    a = jitter_params(VarianceParams(k=3.5), 0.1, np.random.default_rng(7))
    b = jitter_params(VarianceParams(k=3.5), 0.1, np.random.default_rng(7))
    assert a.k == b.k


# -- tier 2: is the azimuthal null calibrated? -----------------------------
def _ring_field(rng, n_rings=40, n_sectors=12, per_cell=200, mu=100.0, sigma=10.0):
    """A perfectly isotropic ring stack: every cell drawn from one N(mu, sigma)."""
    ring = np.repeat(np.arange(n_rings), n_sectors * per_cell)
    sec = np.tile(np.repeat(np.arange(n_sectors), per_cell), n_rings)
    val = rng.normal(mu, sigma, ring.size)
    keep = np.ones(ring.size, bool)
    return ring, sec, val, keep, n_rings, n_sectors


def test_excess_scatter_is_zero_under_isotropy():
    """The null: with no anisotropy the excess must sit at the noise floor.

    `excess` subtracts the variance noise explains before taking a square root
    and clips at zero, so under H0 it is a small POSITIVE residual, not zero:
    Var_b is itself estimated from only S sector means, and half the time it
    lands above its expectation. Measured here (12 sectors, 200 px/cell) that
    sampling floor is ~0.19% of the ring mean, against a raw sector-mean scatter
    of sigma/sqrt(n) = 0.7% -- so the noise subtraction removes ~75% of it, and
    what remains is the resolution limit of the metric. Effects below it are not
    detectable, which is why the study reports the leakage and control terms
    alongside the raw value rather than quoting `excess` on its own.
    """
    rng = np.random.default_rng(0)
    ring, sec, val, keep, nr, ns = _ring_field(rng)
    e, mu, _ = excess_scatter(*cell_moments(ring, sec, val, keep, nr, ns))
    assert np.isfinite(e).all()
    assert np.nanmedian(e) < 0.003, f"null excess {np.nanmedian(e):.4f} is too high"
    assert abs(np.nanmedian(mu) - 100.0) < 1.0


def test_excess_scatter_detects_an_anisotropic_sector():
    """One sector biased by 5% must clear the sampling floor by a wide margin.

    Measured: 1.47% against a 0.19% null, a factor of ~8. That ratio is the
    metric's working dynamic range at this cell size."""
    rng = np.random.default_rng(1)
    ring, sec, val, keep, nr, ns = _ring_field(rng)
    clean = excess_scatter(*cell_moments(ring, sec, val, keep, nr, ns))[0]
    val = val.copy()
    val[sec == 3] *= 1.05
    dirty = excess_scatter(*cell_moments(ring, sec, val, keep, nr, ns))[0]
    assert np.nanmedian(dirty) > 5 * np.nanmedian(clean)
    assert np.nanmedian(dirty) > 0.01


def test_excess_scatter_does_not_grow_with_sample_size():
    """Scale-freeness: the effect size must not depend on pixels per cell, which
    is exactly the property Welch's F lacks (see the module docstring)."""
    vals = []
    for per_cell in (50, 200, 800):
        rng = np.random.default_rng(2)
        ring, sec, val, keep, nr, ns = _ring_field(rng, per_cell=per_cell)
        val = val.copy()
        val[sec == 3] *= 1.05
        vals.append(np.nanmedian(
            excess_scatter(*cell_moments(ring, sec, val, keep, nr, ns))[0]))
    assert max(vals) / min(vals) < 1.5, f"excess drifted with n: {vals}"


# -- tier 3: is the stationarity chi2 calibrated? --------------------------
def _fold_moments(rng, k=10, n_pix=20000, per_fold=80, mu=5.0, gains=None):
    """Synthetic FoldMoments-shaped arrays: Poisson-ish pixels, k folds."""
    from automask.unsupervised.folds import FoldMoments

    gains = np.ones(k) if gains is None else np.asarray(gains)
    shape = (2, 100, n_pix // 200)
    n = np.full(k, float(per_fold))
    s1 = np.empty((k, *shape)); s2 = np.empty((k, *shape))
    for i in range(k):
        x = rng.poisson(mu * gains[i], size=(per_fold, *shape)).astype(float)
        s1[i] = x.sum(axis=0)
        s2[i] = (x ** 2).sum(axis=0)
    return FoldMoments(run=0, n=n, s1=s1, s2=s2,
                       indices=np.arange(k * per_fold),
                       fold_of_shot=np.repeat(np.arange(k), per_fold))


def _chi2_flags(fm, run_geometry):
    """Run `anomaly_field`'s statistic without the assembled-space projection."""
    from scipy import stats as st

    mean, se = fm.fold_means()
    ref = mean.mean(axis=0)
    live = ref > 0
    gain = np.array([np.median(mean[i][live] / ref[live]) for i in range(mean.shape[0])])
    m = mean / gain[:, None, None, None]
    s = se / gain[:, None, None, None]
    w = np.where(s > 0, 1.0 / np.maximum(s, 1e-12) ** 2, 0.0)
    mbar = (w * m).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-300)
    chi2 = (w * (m - mbar) ** 2).sum(axis=0)
    dof = np.maximum((s > 0).sum(axis=0) - 1, 1)
    return st.chi2.sf(chi2, dof), live


def test_stationarity_false_positive_rate_is_near_nominal():
    """The metric reports "fraction of surviving pixels that are anomalous" and
    compares it against the test's own false-positive rate, so that rate has to
    be what it claims. Stationary Poisson pixels, no defects: the rejection rate
    at p < 1e-3 must land within a factor of a few of 1e-3."""
    rng = np.random.default_rng(3)
    fm = _fold_moments(rng)
    p, live = _chi2_flags(fm, None)
    rate = float((p[live] < P_ANOM).mean())
    assert rate < 10 * P_ANOM, f"false-positive rate {rate:.2e} >> nominal {P_ANOM:.0e}"


def test_stationarity_is_blind_to_global_beam_drift():
    """A run whose intensity wanders by 30% must NOT make every pixel anomalous:
    that is what dividing by the per-fold gain is for. Without it this rate goes
    to ~1 and the metric measures the machine instead of the detector."""
    rng = np.random.default_rng(4)
    gains = 1.0 + 0.3 * np.linspace(-1, 1, 10)
    fm = _fold_moments(rng, gains=gains)
    p, live = _chi2_flags(fm, None)
    rate = float((p[live] < P_ANOM).mean())
    assert rate < 10 * P_ANOM, f"beam drift leaked into the statistic ({rate:.2e})"


def test_stationarity_flags_a_pixel_that_changes_mid_run():
    """A pixel whose level jumps halfway through the run must be rejected, even
    though its run-averaged mean and variance are unremarkable."""
    rng = np.random.default_rng(5)
    fm = _fold_moments(rng)
    # double the rate in the second half of the run for one pixel
    fm.s1[5:, 0, 0, 0] *= 2.0
    fm.s2[5:, 0, 0, 0] *= 4.0
    p, _ = _chi2_flags(fm, None)
    assert p[0, 0, 0] < P_ANOM, f"mid-run jump not detected (p = {p[0, 0, 0]:.2e})"


# -- tier 0 ----------------------------------------------------------------
def test_compactness_separates_blobs_from_speckle():
    from automask.unsupervised import parsimony

    class Ctx:
        run = 475
        sample = None
        def __init__(self, floor): self._floor = floor
        def floor(self): return self._floor

    shape = (60, 60)
    floor = np.zeros(shape, bool); floor[0] = True
    rng = np.random.default_rng(6)
    speckle = floor.copy()
    idx = rng.choice(np.arange(59 * 60), size=100, replace=False)
    flat = speckle[1:].ravel(); flat[idx] = True
    speckle[1:] = flat.reshape(59, 60)
    blob = floor.copy(); blob[20:30, 20:30] = True

    c_speck = Candidate("speckle", lambda s: speckle)
    c_blob = Candidate("blob", lambda s: blob)
    assert parsimony.compactness(c_blob, Ctx(floor)) == 1.0
    assert parsimony.compactness(c_speck, Ctx(floor)) < 0.5


def test_plausible_frac_penalises_both_extremes():
    from automask.unsupervised import parsimony

    class Ctx:
        run = 475
        sample = None
        def floor(self): return np.zeros((100, 100), bool)

    def cand(frac):
        m = np.zeros((100, 100), bool)
        m.ravel()[:int(frac * 10000)] = True
        return Candidate(f"f{frac}", lambda s, m=m: m)

    ctx = Ctx()
    assert parsimony.plausible_frac(cand(0.10), ctx) == 1.0
    assert parsimony.plausible_frac(cand(0.001), ctx) < 0.5
    assert parsimony.plausible_frac(cand(0.90), ctx) < 0.5


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    main()
