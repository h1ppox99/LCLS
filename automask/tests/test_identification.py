"""
Tests for the condition-contrast identification package. Same style as
test_unsupervised.py -- plain asserts, no frozen dataset, no psana:

    python -m automask.tests.test_identification

These are property tests, not smoke tests. Each estimator in
`automask.identification` is supposed to have an exact algebraic behaviour --
`J` cancels in a contrast, ring-constant structure is unrecoverable, an ROF
level set is a global segmentation optimum -- and a test that only checks the
shape of the output would pass on an estimator that quietly returns noise. The
cases below build data whose answer is known by construction and check it.
"""
from __future__ import annotations

import numpy as np

from automask.identification import forward as fw
from automask.identification import harmonics as H
from automask.identification import priors as P
from automask.identification import twoway as TW


# -- conditions -------------------------------------------------------------
def test_condition_labels_are_balanced_and_contiguous():
    from automask.identification.conditions import label_shots
    from automask.shot_selection import ShotMeta

    n = 400
    rng = np.random.default_rng(0)
    meta = ShotMeta(run=1, beam_on=np.ones(n, bool),
                    cc_open=rng.random(n) > 0.5, vcc_open=np.zeros(n, bool),
                    intensity={"sample_diode": rng.lognormal(0, 0.3, n)})
    idx = np.arange(n)

    for scheme in ("time", "flux"):
        c = label_shots(meta, idx, scheme, n_groups=4)
        counts = c.counts
        assert c.n_groups == 4, scheme
        assert counts.sum() == n, scheme
        assert counts.max() - counts.min() <= 1, f"{scheme} groups are unbalanced"
        assert set(np.unique(c.labels)) == {0, 1, 2, 3}, scheme

    # flux groups must be ordered in the axis, or "condition" means nothing
    c = label_shots(meta, idx, "flux", n_groups=4)
    assert np.all(np.diff(c.values) > 0), "flux group medians must increase"

    b = label_shots(meta, idx, "branch", n_groups=4)
    assert b.n_groups == 2, "cc on/off with vcc closed is two branch states"


def test_delay_scheme_refuses_to_guess():
    from automask.identification.conditions import label_shots
    from automask.shot_selection import ShotMeta

    meta = ShotMeta(run=1, beam_on=np.ones(8, bool), cc_open=np.ones(8, bool),
                    vcc_open=np.zeros(8, bool),
                    intensity={"sample_diode": np.ones(8)})
    try:
        label_shots(meta, np.arange(8), "delay", 4)
    except ValueError:
        pass
    else:
        raise AssertionError("scheme='delay' must not silently invent an axis")


# -- twoway: identifiability ------------------------------------------------
def test_ring_constant_structure_is_unrecoverable():
    """The core identifiability claim, on a field with no noise at all.

    A shadow that covers a whole ring uniformly is algebraically confounded with
    the sample signal, so a correct estimator must return exactly 1 there -- and
    a compact shadow in the same field must still come back.
    """
    cfg = fw.SynthConfig(read=0.0, photon_scale=0.0, noisy_sem=False)
    t = fw.truth(cfg)
    clean = t.tau[None] * t.A[None] * t.S + t.J[None]
    zbar = np.where(t.valid[None], clean, np.nan)
    sem2 = np.full_like(zbar, 1.0)

    s1 = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                (0, 2), sem2=sem2, q=t.q)
    ring = float(np.median(s1.tau_rel[t.ring_shadow]))
    compact = float(np.median(s1.tau_rel[t.shadow]))
    # Judged against the depth it would have if it WERE recoverable: what is
    # left is the ring detrend's own residual curvature, not the shadow.
    ring_depth = 1.0 - cfg.ring_shadow_transmission
    assert abs(ring - 1.0) < 0.01 * ring_depth, (
        f"a ring-wide shadow must be invisible, got {abs(ring-1):.2e} "
        f"against a true depth of {ring_depth}")
    assert abs(compact - cfg.shadow_transmission) < 0.02, (
        f"a compact shadow must be recovered, got {compact} "
        f"vs {cfg.shadow_transmission}")


def test_contrast_cancels_the_additive_term_exactly():
    """`J` enters both conditions identically, so the difference is free of it."""
    t0 = fw.truth(fw.SynthConfig(streak_amplitude=0.0, read=0.0,
                                 photon_scale=0.0, noisy_sem=False))
    t1 = fw.truth(fw.SynthConfig(streak_amplitude=0.8, read=0.0,
                                 photon_scale=0.0, noisy_sem=False))

    def tau(t):
        z = np.where(t.valid[None], t.tau[None] * t.A[None] * t.S + t.J[None], np.nan)
        return TW.stage1_transmission(z, t.A, t.ring_idx, t.valid, t.n_rings,
                                      (0, 2), sem2=np.ones_like(z), q=t.q).tau_rel

    a, b = tau(t0), tau(t1)
    off = t0.valid & ~t0.streak & np.isfinite(a) & np.isfinite(b)
    assert np.nanmax(np.abs(a[off] - b[off])) < 1e-9, (
        "an 80% streak must not move the transmission estimate anywhere")


def test_design_rank_deficiency_is_one_per_ring():
    d = TW.design_rank_deficiency([5, 5, 5], n_conditions=4)
    assert d["predicted_deficiency"] == 3
    assert d["predicted_rank"] == d["n_params"] - 3


def test_ring_trend_removes_a_radial_slope_the_constant_leaves():
    """The systematic that dominates stage 1 if the ring is reduced to a scalar."""
    rng = np.random.default_rng(1)
    n = 4000
    q = rng.uniform(1.0, 1.1, n)
    y = 10.0 + 50.0 * (q - 1.05) + rng.normal(0, 0.01, n)
    ring_idx = np.zeros(n, dtype=int)
    valid = np.ones(n, bool)

    flat = TW.reduce_rings(y, ring_idx, valid, 1, "median")[ring_idx]
    trend = TW.ring_trend(y, q, ring_idx, valid, 1)
    assert np.std(y - trend) < 0.1 * np.std(y - flat), (
        "a linear-in-q fit must beat a constant on a field with a radial slope")
    assert np.std(y - trend) < 0.02, "and it must reach the noise level"


def test_llr_forms_agree_at_the_nominal_amplitude_and_differ_in_sign():
    a = 2.5
    po = np.exp(-4.0)
    assert abs(TW.llr(np.array([a]), po)[0]
               - TW.llr(np.array([a]), po, amplitude=a)[0]) < 1e-12, (
        "the quadratic is the linear form's tangent at t = a")

    # The operative difference: the matched form is SIGNED.
    low = np.array([-a])
    assert TW.llr(low, po)[0] == TW.llr(np.array([a]), po)[0], (
        "the two-sided form cannot tell a hot pixel from a dead one")
    assert TW.llr(low, po, amplitude=a)[0] < TW.llr(np.array([a]), po, amplitude=a)[0]

    # ...and linear, so region sums accumulate as a matched filter.
    lin = TW.llr(np.array([0.0, 1.0, 2.0]), 0.1, amplitude=a)
    assert np.allclose(np.diff(lin, 2), 0.0, atol=1e-12)


# -- harmonics --------------------------------------------------------------
def test_polarization_is_two_harmonics_in_linear_and_not_in_log():
    """The H1 claim, as an assertion rather than a table."""
    out = H.log_pol_leakage(np.radians(28.9))
    assert out["linear_above_m2"] < 1e-12, (
        "P is exactly m=0 plus m=2 in the linear domain")
    assert out["log_above_m2"] > 1e-4, (
        "log P is NOT band-limited; an analytic m=+-2 removal on the log leaves "
        "a residual")
    # and the leakage must grow with 2theta, i.e. with the depth of the factor
    shallow = H.log_pol_leakage(np.radians(5.0))
    assert shallow["log_above_m2"] < out["log_above_m2"]


def test_tophat_bandwidth_matches_the_2pi_over_width_rule():
    for width_deg in (5.0, 10.0, 20.0):
        w = np.radians(width_deg)
        first_zero = 2 * np.pi / w
        assert H.surviving_fraction(w, int(0.2 * first_zero)) > 0.6, (
            f"a {width_deg} deg shadow must survive a cut well below its first zero")
        assert H.surviving_fraction(w, int(2 * first_zero)) < 0.15, (
            f"and must be gone well above it")


def test_angular_coverage_is_not_fooled_by_the_wrap():
    """Two opposite arcs cover a third of the circle, not all of it."""
    chi = np.concatenate([np.linspace(-0.5, 0.5, 200),
                          np.linspace(np.pi - 0.5, np.pi + 0.5, 200)])
    cov, gap = H.angular_coverage(chi)
    assert abs(np.degrees(cov) - 2 * np.degrees(1.0)) < 5.0, (
        f"two 1-radian arcs cover ~115 deg, got {np.degrees(cov):.0f}")
    assert abs(gap - (np.pi - 1.0)) < 0.05

    full = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    cov_full, _ = H.angular_coverage(full)
    assert cov_full > 0.99 * 2 * np.pi

    arc = np.linspace(0.2, 0.2 + np.radians(100.0), 800)
    cov_arc, _ = H.angular_coverage(arc)
    assert abs(np.degrees(cov_arc) - 100.0) < 5.0


def test_arc_rescaling_restores_conditioning():
    chi = np.linspace(0.2, 0.2 + np.radians(100.0), 1500)
    assert H.max_usable_order(chi, m_limit=16) < 2, (
        "full-circle harmonics are not computable on a 100 deg arc")
    assert H.max_usable_order(chi, m_limit=16, on_arc=True) >= 8, (
        "rescaling the arc to a full period must make them computable")


# -- priors -----------------------------------------------------------------
def test_rof_level_set_beats_perturbations_of_itself():
    """The perimeter MAP claim: `{ROF > 0}` is a global optimum, so no local
    edit of it can lower the energy."""
    rng = np.random.default_rng(4)
    f = rng.normal(-1.0, 1.0, (48, 48))
    f[12:22, 12:22] += 3.0
    lam = 1.0

    def energy(mask):
        return -float(f[mask].sum()) + lam * P.perimeter_cost(mask)

    best = P.perimeter_map(f, lam, n_iter=400)
    e0 = energy(best)
    for _ in range(60):
        m = best.copy()
        i, j = rng.integers(0, 48, 2)
        m[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3] ^= True
        assert energy(m) >= e0 - 1e-6, "found a lower-energy neighbour of the MAP"


def test_perimeter_prior_charges_a_streak_by_its_length():
    """The formal reason one prior cannot cover both classes."""
    blob = np.zeros((64, 64), bool)
    blob[26:38, 26:38] = True                    # 144 px, perimeter 48
    streak = np.zeros((64, 64), bool)
    streak[30:33, 2:50] = True                   # 144 px, perimeter ~102
    assert blob.sum() == streak.sum()
    assert P.perimeter_cost(streak) > 2 * P.perimeter_cost(blob), (
        "equal-area streak must cost far more perimeter than a compact region")


def test_curve_map_recovers_a_line_the_perimeter_map_cannot():
    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 1.0, (120, 120))
    line = np.zeros((120, 120), bool)
    line[58:61, 20:100] = True
    z[line] += 2.5
    llr = TW.llr(z, np.exp(-2.0), amplitude=2.5)

    mask, found = P.curve_map(llr, lam_n=10.0, max_streaks=2)
    recall = float((mask & line).sum()) / int(line.sum())
    assert found, "no line accepted on an obvious streak"
    assert recall > 0.5, f"curve prior recovered only {recall:.2f} of the streak"
    assert float((mask & ~line).sum()) / mask.sum() < 0.5, "and must not smear"


def test_radon_and_line_band_are_inverse_enough_to_localize():
    """The band drawn for a detected peak must land on the feature that made it."""
    field = np.zeros((100, 100))
    field[48:51, 10:90] = 5.0
    angles = np.linspace(0.0, 180.0, 180, endpoint=False)
    sino = P.radon(field, angles)
    a, o = np.unravel_index(int(np.argmax(sino)), sino.shape)
    band, _ = _band_only(field.shape, angles[a], o, 3, field)
    overlap = (band & (field > 0)).sum() / max(band.sum(), 1)
    assert overlap > 0.5, f"back-projected band misses the feature ({overlap:.2f})"


def _band_only(shape, angle, offset, width, llr):
    return P._line_band(shape, angle, offset, width, llr)


def test_blob_bank_whitening_is_scale_free():
    """`max_R l_R` must not prefer large R on pure noise."""
    rng = np.random.default_rng(6)
    t = rng.normal(0.0, 1.0, (200, 200))
    best, arg = P.blob_bank(t, radii=(1, 2, 3, 5, 8))
    counts = np.bincount(arg.ravel(), minlength=9)[[1, 2, 3, 5, 8]]
    frac = counts / counts.sum()
    assert frac.max() < 0.45, (
        f"one radius dominates on pure noise, so the whitening is wrong: {frac}")


# -- forward model ----------------------------------------------------------
def test_simulator_composes_the_two_artifacts_as_the_model_says():
    t = fw.truth(fw.SynthConfig(read=0.0, photon_scale=0.0, noisy_sem=False))
    assert np.all(t.tau[t.shadow] < 1.0), "shadow must be multiplicative and < 1"
    assert np.all(t.J[t.shadow] == 0.0), "classes must not overlap in the truth"
    assert np.all(t.J[t.streak] > 0.0), "streak must be additive and positive"
    assert np.all(t.tau[t.streak] == 1.0)
    # the signal must actually move between conditions, or nothing is identifiable
    spread = np.ptp(t.S[:, t.valid], axis=0) / np.maximum(t.S[0][t.valid], 1e-9)
    assert np.median(spread) > 0.05


def test_simulator_noise_matches_its_own_error_bars():
    zbar, sem2, t = fw.simulate(fw.SynthConfig(noisy_sem=False),
                                rng=np.random.default_rng(9))
    clean = t.tau[None] * t.A[None] * t.S + t.J[None]
    pull = (zbar - clean)[np.broadcast_to(t.valid, zbar.shape)] / \
        np.sqrt(sem2[np.broadcast_to(t.valid, zbar.shape)])
    assert abs(np.mean(pull)) < 0.02, "the draw must be unbiased"
    assert abs(np.std(pull) - 1.0) < 0.02, "and match its stated standard error"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
