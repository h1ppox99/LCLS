"""
Statistical tests for runtime evaluation. Same style as test_components.py --
plain asserts, tiny synthetic arrays, no frozen dataset and no psana:

    python -m automask.tests.test_evaluation_runtime

These are not smoke tests. Each runtime diagnostic asserts a statistical claim
("this is ~0 when the hypothesis holds", "this rejects at the stated rate"), and
a metric whose null is miscalibrated reports defects that are not there. So the
important cases here build data where the truth is known BY CONSTRUCTION -- pure
noise, or noise plus one injected anomaly -- and check the statistic against it.
"""
from __future__ import annotations

import numpy as np

from automask.evaluation.azimuthal import (
    azimuthal_diagnostics, cell_moments, excess_scatter, ring_reference,
)
from automask.evaluation.stability import mask_iou


# -- base ------------------------------------------------------------------
def test_iou_conventions():
    a = np.zeros((8, 8), bool); a[:4] = True
    b = np.zeros((8, 8), bool); b[2:6] = True
    assert abs(mask_iou(a, b) - 2 / 6) < 1e-12
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0
    assert mask_iou(a, a) == 1.0


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


def _noisy_and_biased(rng, noisy_var=8.0, bias=1.04, n_rings=12, n_sectors=12,
                      per_cell=200, mu=100.0, sigma=10.0):
    """A ring stack carrying two DIFFERENT things a mask might remove.

    Sector 3 is azimuthally innocent -- same mean as the rest -- but 8x noisier
    per pixel. Sector 7 carries a real 4% anisotropy. A mask that removes sector
    3 has removed noise, not anisotropy, and the statistic must not reward it by
    inflating.
    """
    ring = np.repeat(np.arange(n_rings), n_sectors * per_cell)
    sec = np.tile(np.repeat(np.arange(n_sectors), per_cell), n_rings)
    val = rng.normal(mu, sigma, ring.size)
    noisy = sec == 3
    val[noisy] = rng.normal(mu, sigma * noisy_var, noisy.sum())
    val[sec == 7] *= bias
    return ring, sec, val, noisy, n_rings, n_sectors


def test_frozen_reference_removes_the_perverse_coupling():
    """Masking noisy-but-isotropic pixels must not INCREASE the excess.

    The candidate-local noise estimate has the defect that sinks Welch's F, in
    milder form: `sigma_hat^2` is pooled over the pixels the candidate left
    behind, so removing a noisy region shrinks the term subtracted from the
    numerator and the statistic rises. Measured here it rises by +1.28 points --
    the mask is penalised for working. With the reference frozen on the
    floor-only pixels the same mask moves the statistic DOWN.
    """
    perverse = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        ring, sec, val, noisy, nr, ns = _noisy_and_biased(rng)
        allpx = np.ones(ring.size, bool)
        ref = ring_reference(*cell_moments(ring, sec, val, allpx, nr, ns))

        def med(keep, **kw):
            mom = cell_moments(ring, sec, val, keep, nr, ns)
            return np.nanmedian(excess_scatter(*mom, **kw)[0])

        perverse += med(~noisy) > med(allpx)
        assert med(~noisy, ref=ref) < med(allpx, ref=ref), \
            f"frozen reference still rewards masking noise (seed {seed})"
    assert perverse >= 4, \
        f"the coupling this test targets showed up in only {perverse}/6 draws"


def test_frozen_reference_stays_sensitive_to_a_real_anisotropy():
    """The freeze must not be bought by clipping everything to zero.

    The floor removes geometry and calib defects but NOT the intensity defects,
    so a frozen `sigma_hat^2` pooled over it inherits them: on this field the
    pooled reduction returns ~644 against a true 100 and every candidate reads
    exactly 0.000%. `ring_reference` reduces robustly instead, which is what
    keeps the injected 4% anisotropy visible.
    """
    rng = np.random.default_rng(3)
    ring, sec, val, noisy, nr, ns = _noisy_and_biased(rng)
    allpx = np.ones(ring.size, bool)
    sig2, _ = ring_reference(*cell_moments(ring, sec, val, allpx, nr, ns))
    assert abs(np.nanmedian(sig2) - 100.0) < 15.0, \
        f"reference noise scale {np.nanmedian(sig2):.0f} is contaminated by the defect"

    e = excess_scatter(*cell_moments(ring, sec, val, allpx, nr, ns),
                       ref=(sig2, np.full(nr, 100.0)))[0]
    assert np.nanmedian(e) > 0.005, "the frozen form clipped the real anisotropy away"
    assert np.mean(e[np.isfinite(e)] == 0) < 0.1, "too many rings clip to zero"


def test_frozen_reference_is_a_no_op_without_a_noise_defect():
    """On a ring with no variance-inflating defect the freeze must change
    nothing -- otherwise it would silently move every number already measured
    against the human masks."""
    rng = np.random.default_rng(4)
    ring, sec, val, _, nr, ns = _noisy_and_biased(rng, noisy_var=1.0)
    allpx = np.ones(ring.size, bool)
    mom = cell_moments(ring, sec, val, allpx, nr, ns)
    ref = ring_reference(*mom)
    a = np.nanmedian(excess_scatter(*mom)[0])
    b = np.nanmedian(excess_scatter(*mom, ref=ref)[0])
    assert abs(a - b) < 1e-4, f"freeze moved the clean case: {a:.6f} vs {b:.6f}"


def test_azimuthal_diagnostics_use_repeated_controls():
    class Frame:
        @staticmethod
        def excess(mask):
            return np.array([1.0, 2.0]) if mask[0] else np.array([3.0, 4.0])

        @staticmethod
        def random_control(mask, rng):
            return np.array([False])

    result = azimuthal_diagnostics(
        Frame(), np.array([True]), np.random.default_rng(0), controls=5)
    assert result["excess"] == 1.5
    np.testing.assert_array_equal(result["gain"], np.full(5, 2.0))
    np.testing.assert_array_equal(result["win_rate"], np.ones(5))


# -- the two fold axes ------------------------------------------------------
def test_dealt_folds_balance_a_clustered_condition():
    """The reason `folds.py` deals shots instead of interleaving blocks.

    A condition that clusters -- as the CC/VCC branch does, in runs of a
    thousand-odd shots -- leaves contiguous blocks with wildly different mixes
    and dealt folds with the same one. Built here as a two-state condition in
    long runs, so the answer is known by construction.
    """
    from automask.evaluation.resampling import assign_folds

    n, k, block = 800, 10, 200
    cond = (np.arange(n) // block) % 2          # long runs, period 2*block
    blocks, dealt = assign_folds(n, k)
    by_block = np.array([cond[blocks == i].mean() for i in range(k)])
    by_deal = np.array([cond[dealt == i].mean() for i in range(k)])
    assert np.ptp(by_block) > 0.9, "the test condition must actually cluster"
    assert np.ptp(by_deal) < 0.05, (
        f"dealt folds must see the same mix, spread was {np.ptp(by_deal):.3f}")


def test_alternating_is_shot_parity_and_halves_stay_chronological():
    from automask.evaluation.resampling import FoldMoments, assign_folds

    n, k = 800, 10
    blocks, dealt = assign_folds(n, k)
    even, odd = FoldMoments.alternating(type("F", (), {"k": k, "n": np.zeros(k)})())
    assert set(dealt[np.isin(dealt, even)] % 2) == {0}, (
        "even dealt folds must be exactly the even-numbered shots")
    # halves() indexes the chronological axis, so it must split the run in time
    first, second = FoldMoments.halves(type("F", (), {"k": k, "n": np.zeros(k)})())
    shots_first = np.flatnonzero(np.isin(blocks, first))
    assert shots_first.max() < np.flatnonzero(np.isin(blocks, second)).min()


def test_moments_axes_partition_the_same_shots():
    rng = np.random.default_rng(0)
    fm = _fold_moments(rng, k=4, n_pix=800, per_fold=20)
    total_block = fm.moments(range(fm.k))[0]
    total_deal = fm.moments(range(fm.k), dealt=True)[0]
    assert total_block == total_deal, "both axes must cover every shot once"


# -- synthetic fold moments ------------------------------------------------
def _fold_moments(rng, k=10, n_pix=20000, per_fold=80, mu=5.0, gains=None):
    """Synthetic FoldMoments-shaped arrays: Poisson-ish pixels, k folds."""
    from automask.evaluation.resampling import FoldMoments

    gains = np.ones(k) if gains is None else np.asarray(gains)
    shape = (2, 100, n_pix // 200)
    n = np.full(k, float(per_fold))
    s1 = np.empty((k, *shape)); s2 = np.empty((k, *shape))
    for i in range(k):
        x = rng.poisson(mu * gains[i], size=(per_fold, *shape)).astype(float)
        s1[i] = x.sum(axis=0)
        s2[i] = (x ** 2).sum(axis=0)
    # Tier 3 reads the chronological axis only; the dealt arrays are carried so
    # the object is well formed, and a per-fold gain does not survive a deal.
    return FoldMoments(run=0, n=n, s1=s1, s2=s2,
                       indices=np.arange(k * per_fold),
                       block_of_shot=np.repeat(np.arange(k), per_fold),
                       dn=n.copy(), ds1=s1.copy(), ds2=s2.copy(),
                       fold_of_shot=np.tile(np.arange(k), per_fold))


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    main()
