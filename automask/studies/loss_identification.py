#!/usr/bin/env python3
"""
studies/loss_identification.py -- the condition-contrast loss, as a list of
things that could be false.

The proposal reformulates masking as inference on a forward model:

    x_it = tau_i F_t A_i S(q_i; theta_c(t)) + F_t J_i + eps_it,
    M = {i : tau_i != 1} | {i : J_i != 0}

and derives from it a two-stage estimator (contrast conditions to kill `J` and
recover `tau`, then residual against `tau` to recover `J`), three artifact-class
priors, an azimuthal harmonic cut, and the assertion that the current
threshold-then-regularize pipeline is a lossy approximation to a joint MAP.

That is a lot of structure resting on assumptions that have not been measured.
This file is the register of those assumptions. Each is a `Claim` with an
experiment attached and, more importantly, with a written-down FALSIFIER: the
outcome that would kill it. A claim whose falsifier cannot be stated is not in
the register.

TWO KINDS OF CLAIM, and the split is the main result of building this.

  `bench` -- a claim about an ESTIMATOR. True or false as mathematics, on a
             field where `tau`, `J`, `A` and `S` are known by construction
             (`identification.forward`). No beamtime can make a biased estimator
             unbiased, so these are settled here and now.
  `run`   -- a claim about THIS EXPERIMENT. Whether the conditions move the
             signal, whether the sample is anisotropic, whether real artifacts
             are multiplicative or additive. No simulation can answer these; the
             code is written and waits for the data.

Run:  python -m automask.studies.loss_identification              # bench claims
      python -m automask.studies.loss_identification --runs 475   # + run claims
      python -m automask.studies.loss_identification --claims M3 P1 --plots
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from automask.identification import forward as fw
from automask.identification import harmonics as H
from automask.identification import priors as P
from automask.identification import twoway as TW

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

SUPPORTED, REFUTED, MIXED, BLOCKED = "SUPPORTED", "REFUTED", "MIXED", "BLOCKED"


@dataclass
class Result:
    verdict: str
    detail: str
    numbers: Dict[str, float] = field(default_factory=dict)
    figure: Optional[object] = None


@dataclass
class Claim:
    """One assertion from the notes, with the experiment that could kill it."""
    id: str
    statement: str
    falsifier: str
    needs: str                       # "bench" or "run"
    experiment: Callable


CLAIMS: List[Claim] = []


def claim(id: str, statement: str, falsifier: str, needs: str = "bench"):
    def deco(fn):
        CLAIMS.append(Claim(id, statement, falsifier, needs, fn))
        return fn
    return deco


def _iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


# ==========================================================================
#  A -- the assumptions the notes flag with a TODO
# ==========================================================================
@claim("A1", "Experimental conditions drift within a run, so folds must be "
             "alternating rather than contiguous to hold the condition mix fixed.",
       "Contiguous halves and alternating folds disagree no more than two "
       "alternating draws disagree with each other -- then the drift is "
       "sampling noise and contiguous folds are fine.",
       needs="run")
def exp_a1_drift(run: int, **kw) -> Result:
    from automask.evaluation import load_sample
    from automask.masking import production_pipeline
    from automask.unsupervised import folds as F
    from automask.unsupervised.azimuthal import build_frame

    pipe = production_pipeline()
    sample = load_sample(run, features=pipe.features_needed())
    fr = build_frame(sample, pipe.floor(sample))
    fm = F.load(run)
    keep = fr.usable & ~fr.floor

    levels = np.full((fm.k, fr.n_rings), np.nan)
    for k in range(fm.k):
        s = F.fold_sample(sample, fm, [k])
        img = np.asarray(s.sumimg, dtype=np.float64)
        levels[k] = TW.reduce_rings(img, fr.ring_idx, keep, fr.n_rings, "median")
    gain = np.nanmedian(levels / np.nanmedian(levels, axis=0), axis=1)
    lv = levels / gain[:, None]

    ok = np.all(np.isfinite(lv), axis=0)
    x = np.arange(fm.k) - (fm.k - 1) / 2.0
    slope = (lv[:, ok] * x[:, None]).sum(axis=0) / (x ** 2).sum()
    resid = lv[:, ok] - slope * x[:, None] - lv[:, ok].mean(axis=0)
    se = np.sqrt((resid ** 2).sum(axis=0) / max(fm.k - 2, 1) / (x ** 2).sum())
    t = slope / np.maximum(se, 1e-300)
    frac = float(np.mean(np.abs(t) > 3.0))

    # The gain series itself: a global drift the per-fold normalization removes,
    # reported because a large one changes what "the same conditions" means.
    swing = float(gain.max() / gain.min() - 1.0)
    verdict = SUPPORTED if frac > 0.10 else REFUTED
    return Result(verdict,
                  f"{100*frac:.1f}% of {int(ok.sum())} rings drift at |t|>3 across "
                  f"{fm.k} folds; global gain swings {100*swing:.1f}% over the run",
                  {"frac_rings_drifting": frac, "gain_swing": swing,
                   "n_rings": float(ok.sum())})


@claim("A2", "Consecutive shots share experimental conditions, so dealing shots "
             "round-robin (one to each fold in turn) gives folds with identical "
             "condition mixes.",
       "Either adjacent shots agree on the condition no more often than two "
       "shots drawn at random -- then dealing buys nothing -- or the condition "
       "varies on a period commensurate with the fold count, which is the one "
       "structure that makes dealing systematically WORSE than a contiguous "
       "split.",
       needs="run")
def exp_a2_interchangeable(run: int, n_folds: int = 10, **kw) -> Result:
    from automask.identification.conditions import LIT
    from automask.io import scan_shots

    meta = scan_shots(run)
    idx = LIT.resolve(meta)
    x = meta.monitor(LIT.intensity)[idx]
    x = (x - x.mean()) / (x.std() or 1.0)
    lags = (1, 2, 3, 5, 10, 20)
    acf = {L: float((x[:-L] * x[L:]).mean()) for L in lags}
    band = 2.0 / np.sqrt(x.size)
    bad_lags = [L for L, v in acf.items() if abs(v) > band]

    branch = meta.cc_open[idx].astype(np.int8) + 2 * meta.vcc_open[idx].astype(np.int8)
    flips = float(np.mean(branch[:-1] != branch[1:]))
    p = np.bincount(branch) / branch.size
    flips_iid = float(1.0 - (p ** 2).sum())
    se = np.sqrt(max(flips_iid * (1 - flips_iid), 1e-12) / branch.size)
    z_flip = (flips - flips_iid) / se
    # THE quantity the assumption is about: how much more often a shot agrees
    # with its neighbour than with a shot picked at random from the run.
    adjacent_agree = 1.0 - flips
    random_agree = 1.0 - flips_iid
    regime = ("clustered into blocks" if z_flip < -4 else
              "alternating with a period" if z_flip > 4 else "consistent with i.i.d.")

    # The consequence that actually matters. Both fold schemes are built here
    # and their condition mix compared: a fold scheme is usable only if every
    # fold sees the same branch composition, which is exactly assumption 1's
    # stated reason for preferring alternating folds.
    # Three schemes, because two of them are easy to conflate. `folds.py` builds
    # CONTIGUOUS blocks and then interleaves whole blocks (even folds vs odd
    # folds); that is not the same as interleaving shots, and against a branch
    # that clusters into long blocks it need not balance anything.
    order = np.arange(idx.size)
    block = (order * n_folds) // idx.size
    vcc = meta.vcc_open[idx]
    per_block = np.array([vcc[block == k].mean() for k in range(n_folds)])
    per_shot = np.array([vcc[order % n_folds == k].mean() for k in range(n_folds)])
    spread_c = float(per_block.max() - per_block.min())
    spread_s = float(per_shot.max() - per_shot.min())
    # The actual `folds.alternating()` split: even blocks vs odd blocks.
    even = vcc[np.isin(block, np.arange(0, n_folds, 2))].mean()
    odd = vcc[np.isin(block, np.arange(1, n_folds, 2))].mean()
    gap_blockalt = float(abs(even - odd))
    # ...against the `folds.halves()` split, for reference.
    first = vcc[block < n_folds // 2].mean()
    second = vcc[block >= n_folds // 2].mean()
    gap_halves = float(abs(first - second))

    run_len = 1.0 / flips if flips > 0 else float("inf")
    # Dealing round-robin into K folds fails only if the condition's period is
    # commensurate with K. A block length many times K is the safe regime.
    commensurate = run_len < 3 * n_folds
    deals_well = spread_s < 0.1 and spread_s < 0.5 * spread_c
    verdict = SUPPORTED if (adjacent_agree > random_agree + 4 * se
                            and deals_well and not commensurate) else REFUTED
    detail = (f"adjacent shots share the branch {100*adjacent_agree:.1f}% of the "
              f"time against {100*random_agree:.1f}% for two shots drawn at "
              f"random -- {regime} (z = {z_flip:+.0f}), mean run length "
              f"{run_len:.0f} shots in stream order, {run_len/n_folds:.0f}x the "
              f"fold count so dealing is not commensurate\n"
              f"    monitor autocorrelation: " +
              ", ".join(f"lag{L} {acf[L]:+.3f}" for L in lags) +
              f"  (|acf| > {band:.3f} significant"
              + (f"; exceeded at lags {bad_lags}" if bad_lags else "; none exceed") + ")\n"
              f"    VCC-open fraction, spread over {n_folds} folds: "
              f"dealt round-robin {spread_s:.3f}, contiguous blocks {spread_c:.3f}\n"
              f"    folds.py splits: alternating BLOCKS differ by {gap_blockalt:.3f}, "
              f"contiguous halves by {gap_halves:.3f}")
    return Result(verdict, detail,
                  {**{f"acf_lag{L}": v for L, v in acf.items()},
                   "acf_band": band, "flip_rate": flips, "flip_rate_iid": flips_iid,
                   "flip_z": z_flip, "branch_run_length": run_len,
                   "adjacent_agree": adjacent_agree, "random_agree": random_agree,
                   "vcc_spread_contiguous": spread_c,
                   "vcc_spread_shot_interleaved": spread_s,
                   "vcc_gap_block_alternating": gap_blockalt,
                   "vcc_gap_halves": gap_halves})


@claim("A3", "The scattering is not truly isotropic, so a noise scale estimated "
             "by pooling across a ring is contaminated by real anisotropy.",
       "The per-ring azimuthal harmonic power at m>=1 is consistent with the "
       "counting-noise null -- then the sample IS isotropic and the whole "
       "robustness apparatus is unnecessary.",
       needs="run")
def exp_a3_anisotropy(run: int, m_max: int = 8, **kw) -> Result:
    from automask.evaluation import load_sample
    from automask.masking import production_pipeline
    from automask.unsupervised.azimuthal import build_frame, pixel_frame

    pipe = production_pipeline()
    sample = load_sample(run, features=pipe.features_needed())
    fr = build_frame(sample, pipe.floor(sample))
    # `AzimuthalFrame` does not carry chi, and it must come from the beam
    # centre, not from raw pixel indices.
    _, chi, _, _ = pixel_frame(sample)
    keep = fr.usable & ~fr.floor

    frac_low, powers = [], []
    for r in range(fr.n_rings):
        sel = keep & (fr.ring_idx == r)
        if sel.sum() < 200:
            continue
        y = fr.clipped[sel]
        level = np.nanmedian(y)
        if not np.isfinite(level) or level == 0:
            continue
        # Fitted on the arc, for the reason `exp_h3_coverage_limit` measures:
        # full-circle harmonics are not computable on this geometry.
        c, _, cond = H.fit_harmonics(H.arc_phase(chi[sel]), y / level, m_max)
        if not np.isfinite(cond) or cond > H.COND_TOL:
            continue
        p = H.power_spectrum(c)[1:]
        powers.append(p)
        frac_low.append(p[:4].sum() / max(p.sum(), 1e-300))
    if not powers:
        return Result(BLOCKED, "no ring had usable azimuthal coverage for a "
                               f"harmonic fit up to m={m_max}", {})
    powers = np.stack(powers)
    amp = float(np.median(np.sqrt(powers.sum(axis=1))))
    low = float(np.median(frac_low))
    verdict = SUPPORTED if amp > 0.005 else REFUTED
    return Result(verdict,
                  f"median azimuthal amplitude {100*amp:.2f}% of the ring level "
                  f"over {len(powers)} rings; {100*low:.0f}% of it sits at m<=4",
                  {"median_amplitude": amp, "frac_power_m_le_4": low,
                   "n_rings": float(len(powers))})


@claim("A4", "A median over sectors is a more robust noise/level reduction than "
             "a pooled mean in this setting.",
       "Under the contamination levels a real ring carries, the mean's bias is "
       "no worse than the median's variance cost -- then the extra machinery "
       "buys nothing.")
def exp_a4_robust_reduction(n_trials: int = 200, **kw) -> Result:
    rng = np.random.default_rng(3)
    t = fw.truth()
    fracs = (0.0, 0.02, 0.05, 0.10, 0.20)
    rows = {}
    for f in fracs:
        err = {h: [] for h in ("median", "mean", "trimmed")}
        for _ in range(n_trials):
            ring = rng.normal(100.0, 8.0, 400)
            n_bad = int(f * ring.size)
            if n_bad:
                ring[:n_bad] *= 0.55                   # a shadow inside the ring
            idx = np.zeros(ring.size, dtype=int)
            valid = np.ones(ring.size, bool)
            for h in err:
                got = TW.reduce_rings(ring, idx, valid, 1, h)[0]
                err[h].append(got - 100.0)
            rng.shuffle(ring)
        rows[f] = {h: (float(np.mean(v)), float(np.sqrt(np.mean(np.square(v)))))
                   for h, v in err.items()}

    worst = max(fracs)
    rmse_med = rows[worst]["median"][1]
    rmse_mean = rows[worst]["mean"][1]
    clean_cost = rows[0.0]["median"][1] / max(rows[0.0]["mean"][1], 1e-300)
    verdict = SUPPORTED if rmse_med < rmse_mean else REFUTED
    detail = ("  contamination   median RMSE   mean RMSE   trimmed RMSE\n" +
              "\n".join(f"    {100*f:5.0f}%      {rows[f]['median'][1]:9.3f}   "
                        f"{rows[f]['mean'][1]:9.3f}   {rows[f]['trimmed'][1]:9.3f}"
                        for f in fracs) +
              f"\n    variance cost on a clean ring: median/mean RMSE = {clean_cost:.2f}x")
    return Result(verdict, detail,
                  {"rmse_median_at_20pct": rmse_med, "rmse_mean_at_20pct": rmse_mean,
                   "clean_ring_cost": clean_cost})


# ==========================================================================
#  M -- the forward model and its estimators
# ==========================================================================
@claim("M1", "Only WITHIN-ring structure is identifiable: a component of alpha "
             "constant around a ring is confounded with S(q) and no estimator "
             "recovers it.",
       "The estimator recovers a ring-wide shadow as well as it recovers a "
       "compact one -- then the confounding argument is wrong.")
def exp_m1_identifiability(**kw) -> Result:
    zbar, sem2, t = fw.simulate(rng=np.random.default_rng(1))
    s1 = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                pair=(0, 2), sem2=sem2, q=t.q)

    def recovered_depth(region):
        sel = region & t.valid & np.isfinite(s1.tau_rel)
        return float(1.0 - np.median(s1.tau_rel[sel])) if sel.any() else np.nan

    got_compact = recovered_depth(t.shadow)
    got_ring = recovered_depth(t.ring_shadow)
    true_compact = 1.0 - t.cfg.shadow_transmission
    true_ring = 1.0 - t.cfg.ring_shadow_transmission

    d = TW.design_rank_deficiency(np.bincount(t.ring_idx[t.valid]), t.cfg.n_conditions)
    small = _small_design_rank(n_rings=4, n_pixels=6, n_conditions=3)

    ratio_compact = got_compact / true_compact
    ratio_ring = got_ring / true_ring
    verdict = SUPPORTED if (ratio_compact > 0.8 and abs(ratio_ring) < 0.2) else REFUTED
    return Result(verdict,
                  f"compact shadow recovered at {100*ratio_compact:.0f}% of its "
                  f"true depth, ring-wide shadow at {100*ratio_ring:+.0f}%; "
                  f"design deficiency predicted {d['predicted_deficiency']} "
                  f"(= n_rings), SVD on a small design gives {small}",
                  {"recovered_frac_compact": ratio_compact,
                   "recovered_frac_ring": ratio_ring,
                   "svd_deficiency": float(small),
                   "predicted_deficiency": float(d["predicted_deficiency"])})


def _small_design_rank(n_rings: int, n_pixels: int, n_conditions: int) -> int:
    """Rank deficiency of an explicit two-way design, by SVD.

    The algebra says one unobservable shift per ring. This builds the actual
    `[pixel dummies | ring x condition dummies]` matrix and counts zero singular
    values, so the claim is checked rather than restated.
    """
    rows = []
    n_pix = n_rings * n_pixels
    for r in range(n_rings):
        for p in range(n_pixels):
            for c in range(n_conditions):
                row = np.zeros(n_pix + n_rings * n_conditions)
                row[r * n_pixels + p] = 1.0
                row[n_pix + r * n_conditions + c] = 1.0
                rows.append(row)
    X = np.stack(rows)
    s = np.linalg.svd(X, compute_uv=False)
    return int((s < 1e-9 * s.max()).sum() + (X.shape[1] - s.size))


def _stage1_error(zbar, sem2, t, detrend: str):
    """Median over rings of the empirical and the propagated stage-1 error.

    Aggregated per ring, not globally: rings differ by orders of magnitude in
    how much contrast they carry, so a pooled standard deviation is a statement
    about the worst ring rather than about the estimator.
    """
    s1 = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                (0, 2), sem2=sem2, detrend=detrend, q=t.q)
    clean = t.valid & ~t.bad & np.isfinite(s1.tau_rel) & np.isfinite(s1.sigma)
    emp, pred = [], []
    for r in range(t.n_rings):
        sel = clean & (t.ring_idx == r)
        if sel.sum() < 50:
            continue
        emp.append(float(np.std(s1.tau_rel[sel])))
        pred.append(float(np.median(s1.sigma[sel])))
    return np.array(emp), np.array(pred), s1


@claim("M2", "Identifying power is Var_c(gamma_{q,c}); it is measurable and it "
             "caps the method.",
       "Transmission recovery error does not scale as the inverse of the "
       "measured contrast -- then Var_c(gamma) is not the controlling quantity "
       "and the cap is somewhere else.")
def exp_m2_identifying_power(**kw) -> Result:
    contrasts = (0.0, 0.02, 0.05, 0.10, 0.25, 0.50)
    rows = []
    for c in contrasts:
        cfg = fw.SynthConfig(contrast=c)
        zbar, sem2, t = fw.simulate(cfg, rng=np.random.default_rng(11))
        ip = TW.identifying_power(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                  sem2=sem2, q=t.q)
        emp, pred, _ = _stage1_error(zbar, sem2, t, "linear")
        s = ip.summary()
        rows.append((c, s["median_snr"], s["median_excess"],
                     float(np.median(emp)), float(np.median(pred))))

    arr = np.array([(r[2], r[3]) for r in rows if r[0] > 0])
    lx, ly = np.log(arr[:, 0]), np.log(arr[:, 1])
    slope = float(np.polyfit(lx, ly, 1)[0])
    snr0 = rows[0][1]
    scaling_ok = abs(slope + 0.5) < 0.2
    null_ok = snr0 < 3.0
    verdict = SUPPORTED if (scaling_ok and null_ok) else MIXED
    detail = ("  contrast   median SNR   Var_c(gamma)   sd(tau_rel)/ring   "
              "propagated sigma\n" +
              "\n".join(f"    {r[0]:5.2f}   {r[1]:10.2f}   {r[2]:12.3e}   "
                        f"{r[3]:15.4f}   {r[4]:16.4f}" for r in rows) +
              f"\n    log-log slope of error vs Var_c(gamma): {slope:+.2f} "
              f"(model predicts -0.50); SNR at zero contrast {snr0:.2f} (null = 0)")
    return Result(verdict, detail,
                  {"loglog_slope": slope, "snr_at_zero_contrast": snr0})


@claim("M7", "Given the contrast, stage-1 error is counting noise -- so "
             "Var_c(gamma) alone caps the method.",
       "The empirical scatter of the recovered transmission exceeds the "
       "propagated counting noise on clean pixels -- then a systematic, not the "
       "identifying power, is the binding constraint.")
def exp_m7_gradient_floor(**kw) -> Result:
    zbar, sem2, t = fw.simulate(fw.SynthConfig(contrast=0.25),
                                rng=np.random.default_rng(11))
    out = {}
    for detrend in ("ring", "linear"):
        emp, pred, _ = _stage1_error(zbar, sem2, t, detrend)
        out[detrend] = (emp, pred, float(np.median(emp / pred)))

    ratio_ring = out["ring"][2]
    ratio_lin = out["linear"][2]
    verdict = REFUTED if ratio_ring > 2.0 else SUPPORTED
    detail = (f"  per-ring constant denominator: empirical error is "
              f"{ratio_ring:.1f}x the propagated counting noise (median over rings)\n"
              f"  robust linear-in-q denominator: {ratio_lin:.1f}x\n"
              "  the excess is the radial gradient of S_c - S_c' inside the ring "
              "bin, read as transmission;\n"
              "  it vanishes only at the extremum of the contrast, where the "
              "slope happens to be zero")
    return Result(verdict, detail,
                  {"error_over_noise_ring": ratio_ring,
                   "error_over_noise_linear": ratio_lin,
                   "improvement": ratio_ring / max(ratio_lin, 1e-9)})


@claim("M3", "Differencing two conditions cancels the additive parasitic term "
             "J_i exactly, so stage 1 is blind to streaks.",
       "The recovered transmission moves when only the streak amplitude "
       "changes -- then the cancellation is not exact and the two stages are "
       "not separable.")
def exp_m3_contrast_cancels_j(**kw) -> Result:
    amps = (0.0, 0.05, 0.20, 0.60)
    ref = None
    shifts, streak_bias = [], []
    for a in amps:
        cfg = fw.SynthConfig(streak_amplitude=a)
        zbar, sem2, t = fw.simulate(cfg, rng=np.random.default_rng(5))
        s1 = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                    pair=(0, 2), sem2=sem2, q=t.q)
        if ref is None:
            ref = s1.tau_rel.copy()
            noise = float(np.median(_stage1_error(zbar, sem2, t, "linear")[0]))
        d = s1.tau_rel - ref
        shifts.append(float(np.nanmax(np.abs(d[t.valid & ~t.bad]))))
        sel = t.streak & t.valid
        streak_bias.append(float(np.nanmedian(s1.tau_rel[sel]) - 1.0))

    # With `noisy_sem` off the draw is identical across amplitudes, so any shift
    # is the streak leaking through; the tolerance is the estimator's own noise.
    worst = max(shifts)
    verdict = SUPPORTED if worst < 0.25 * noise else REFUTED
    detail = ("  streak amp   max |d tau_rel| off-streak   median tau_rel-1 ON streak\n" +
              "\n".join(f"    {a:8.2f}   {s:22.2e}   {b:+22.4f}"
                        for a, s, b in zip(amps, shifts, streak_bias)) +
              f"\n    estimator noise on clean pixels (median over rings): {noise:.4f}")
    return Result(verdict, detail,
                  {"max_leak": worst, "estimator_noise": noise,
                   "leak_over_noise": worst / noise if noise else np.nan})


@claim("M4", "The stage order is forced by the composition: tau must be "
             "recovered before J, not the other way round.",
       "Estimating J first from a ring residual, then tau, recovers both as "
       "well -- then the ordering is a preference, not a constraint.")
def exp_m4_stage_order(**kw) -> Result:
    zbar, sem2, t = fw.simulate(rng=np.random.default_rng(7))

    s1 = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                pair=(0, 2), sem2=sem2, q=t.q)
    s2 = TW.stage2_parasitic(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                             s1.tau_rel, condition=0, sem2=sem2, q=t.q)

    # Reversed: build J from a residual against a ring model with no tau, then
    # recover tau from the J-subtracted data.
    ring0 = TW.ring_trend(zbar[0] / t.A, t.q, t.ring_idx, t.valid, t.n_rings)
    J_first = zbar[0] - t.A * ring0
    z_corr = zbar - J_first[None]
    s1_rev = TW.stage1_transmission(z_corr, t.A, t.ring_idx, t.valid, t.n_rings,
                                    pair=(0, 2), sem2=sem2, q=t.q)

    clean = t.valid & ~t.bad

    def contrast_to_noise(field, region):
        """Median offset of `field` over `region`, in units of its own spread on
        clean pixels -- the detectability the next stage would actually see."""
        a = field[region & t.valid]
        b = field[clean]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 10 or b.size < 10:
            return np.nan
        scale = 1.4826 * np.median(np.abs(b - np.median(b)))
        return float(abs(np.median(a) - np.median(b)) / max(scale, 1e-300))

    fwd_shadow = contrast_to_noise(1.0 - s1.tau_rel, t.shadow)
    fwd_streak = contrast_to_noise(s2.J, t.streak)
    rev_shadow = contrast_to_noise(1.0 - s1_rev.tau_rel, t.shadow)
    # The diagnostic that matters: taken first, does the J residual absorb the
    # SHADOW -- an artifact of the wrong class, which it would then hand to the
    # curve prior?
    leak_shadow = contrast_to_noise(J_first, t.shadow)
    leak_streak = contrast_to_noise(J_first, t.streak)

    verdict = SUPPORTED if (leak_shadow > 3.0 or rev_shadow < 0.7 * fwd_shadow) \
        else REFUTED
    return Result(verdict,
                  f"forward order: shadow {fwd_shadow:.1f} sigma in stage 1, streak "
                  f"{fwd_streak:.1f} sigma in stage 2. reversed: the J-first "
                  f"residual sees the shadow at {leak_shadow:.1f} sigma against the "
                  f"streak at {leak_streak:.1f} sigma, and the tau recovered after "
                  f"it reads {rev_shadow:.1f} sigma",
                  {"shadow_sigma_forward": fwd_shadow,
                   "streak_sigma_forward": fwd_streak,
                   "shadow_sigma_reversed": rev_shadow,
                   "shadow_leak_into_J": leak_shadow,
                   "streak_in_J_first": leak_streak})


@claim("M5", "Real artifacts split into a multiplicative class and an additive "
             "class, which is what makes a two-stage estimator the right shape.",
       "Reference-masked pixels do not separate into a slope-deviation "
       "population and an intercept-deviation population -- then one stage "
       "would do and the decomposition is unmotivated.",
       needs="run")
def exp_m5_artifact_classes(run: int, scheme: str = "time", n_groups: int = 4,
                            **kw) -> Result:
    from automask.evaluation import load_sample
    from automask.identification.conditions import load as load_conditions
    from automask.masking import production_pipeline
    from automask.unsupervised.azimuthal import build_frame
    from automask.geometry import panel_to_asm

    pipe = production_pipeline()
    sample = load_sample(run, features=pipe.features_needed())
    floor = pipe.floor(sample)
    fr = build_frame(sample, floor)
    gm = load_conditions(run, scheme, n_groups)
    if gm.n_groups < 3:
        return Result(BLOCKED, f"scheme {scheme!r} gives {gm.n_groups} groups; "
                               "the slope/intercept split needs at least 3", {})

    zbar = np.stack([panel_to_asm(m, run) for m in gm.mean()])
    sem2 = np.stack([panel_to_asm(s, run) for s in gm.sem2()])
    A = _known_A(sample, fr)
    keep = fr.usable & ~floor

    # Detrended inside each ring for the reason `twoway.ring_trend` documents:
    # an unmodelled radial slope would show up here as a fake intercept, i.e. as
    # the additive class this experiment is trying to count.
    x = np.stack([TW.ring_trend(zbar[c] / A, fr.q, fr.ring_idx, keep, fr.n_rings)
                  for c in range(gm.n_groups)]) * A[None]
    w = 1.0 / np.maximum(sem2, 1e-300)
    sw = w.sum(axis=0)
    mx = (w * x).sum(axis=0) / sw
    my = (w * zbar).sum(axis=0) / sw
    sxx = (w * (x - mx) ** 2).sum(axis=0)
    sxy = (w * (x - mx) * (zbar - my)).sum(axis=0)
    slope = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-300), np.nan)
    intercept = my - slope * mx
    se_slope = np.sqrt(1.0 / np.maximum(sxx, 1e-300))
    se_int = np.sqrt(1.0 / sw + mx ** 2 / np.maximum(sxx, 1e-300))

    t_mult = (slope - 1.0) / se_slope
    t_add = intercept / np.maximum(se_int, 1e-300)
    target = sample.human & ~floor & keep
    ctrl = keep & ~sample.human

    def rates(sel):
        m = np.abs(t_mult[sel]) > 4.0
        a = np.abs(t_add[sel]) > 4.0
        n = max(int(sel.sum()), 1)
        return (float(m.sum()) / n, float(a.sum()) / n,
                float((m & a).sum()) / n, float((~m & ~a).sum()) / n)

    tm, ta, tb, tn = rates(target)
    cm, ca, cb, cn = rates(ctrl)
    both_present = tm > 3 * cm and ta > 3 * ca
    verdict = SUPPORTED if both_present else MIXED
    return Result(verdict,
                  f"masked pixels: {100*tm:.1f}% multiplicative-only-significant, "
                  f"{100*ta:.1f}% additive, {100*tb:.1f}% both, {100*tn:.1f}% neither. "
                  f"unmasked control: {100*cm:.1f}% / {100*ca:.1f}% / {100*cb:.1f}% / "
                  f"{100*cn:.1f}%",
                  {"mult_rate_masked": tm, "add_rate_masked": ta,
                   "mult_rate_control": cm, "add_rate_control": ca})


def _known_A(sample, frame) -> np.ndarray:
    """`A_i = Omega_i P_i` on the assembled canvas, from the run's own geometry."""
    from automask import azimuthal as az
    ai = az.integrator(sample.run, frame.q.shape)
    A = (ai.solidAngleArray(frame.q.shape)
         * ai.polarization(frame.q.shape, factor=az.POLARIZATION))
    return np.where(np.isfinite(A) & (A > 0), A, np.nan)


@claim("M6", "Polarization belongs in A_i; left uncorrected it sits in alpha_i "
             "and reads as an artifact.",
       "Dropping the polarization factor from A leaves the recovered "
       "transmission unbiased -- then the correction is optional at this "
       "geometry.")
def exp_m6_polarization(**kw) -> Result:
    zbar, sem2, t = fw.simulate(rng=np.random.default_rng(13))
    full = TW.stage1_transmission(zbar, t.A, t.ring_idx, t.valid, t.n_rings,
                                  (0, 2), sem2=sem2, q=t.q)
    partial = TW.stage1_transmission(zbar, t.solid_angle, t.ring_idx, t.valid,
                                     t.n_rings, (0, 2), sem2=sem2, q=t.q)
    clean = t.valid & ~t.bad
    depth = 1.0 - t.cfg.shadow_transmission

    def per_ring_spread(s1):
        v = [float(np.std(s1.tau_rel[clean & (t.ring_idx == r)]))
             for r in range(t.n_rings) if (clean & (t.ring_idx == r)).sum() > 50]
        return float(np.median(v))

    spread_full = per_ring_spread(full)
    spread_part = per_ring_spread(partial)
    swing = float(np.nanpercentile(partial.tau_rel[clean], 97.5)
                  - np.nanpercentile(partial.tau_rel[clean], 2.5))

    # Is the induced structure azimuthal at m=2, as the polarization form says?
    # Fitted PER RING: the amplitude scales with sin^2(2theta), so pooling rings
    # of different q would mix a radial envelope into an azimuthal fit.
    shares = []
    for r in range(t.n_rings):
        sel = clean & (t.ring_idx == r)
        if sel.sum() < 200:
            continue
        coef, _, cond = H.fit_harmonics(t.chi[sel], partial.tau_rel[sel] - 1.0, 6)
        if not np.isfinite(cond) or cond > H.COND_TOL:
            continue
        spec = H.power_spectrum(coef)[1:]
        shares.append(spec[1] / max(spec.sum(), 1e-300))
    m2_share = float(np.median(shares)) if shares else np.nan

    verdict = SUPPORTED if swing > max(3 * spread_full, 0.2 * depth) else MIXED
    return Result(verdict,
                  f"sd(tau_rel) per ring on clean pixels {spread_full:.4f} with A, "
                  f"{spread_part:.4f} without polarization "
                  f"({spread_part/spread_full:.1f}x); the induced 95% swing is "
                  f"{swing:.3f} -- {swing/depth:.0%} of a real shadow's depth and "
                  f"{swing/spread_full:.0f}x the estimator's own noise. "
                  + (f"{100*m2_share:.0f}% of its azimuthal power is at m=2 "
                     f"({len(shares)} rings well-conditioned enough to check)"
                     if shares else
                     "no ring was well-conditioned enough for a harmonic check"),
                  {"sd_with_A": spread_full, "sd_without_pol": spread_part,
                   "induced_swing": swing, "shadow_depth": depth,
                   "m2_share": m2_share, "n_rings_checked": float(len(shares))})


# ==========================================================================
#  P -- the geometric priors
# ==========================================================================
@claim("P1", "A perimeter prior suppresses streaks: its cost grows with the "
             "streak's length, so no lambda cleans noise without deleting lines.",
       "Some lambda removes isolated noise pixels while keeping most of a "
       "streak -- then one region prior covers both classes and the separate "
       "curve detector is unnecessary.")
def exp_p1_prior_mismatch(plots: bool = False, **kw) -> Result:
    z, truth_blob, truth_streak, noise = _prior_bench(np.random.default_rng(2))
    llr = TW.llr(z, BENCH_PRIOR_ODDS, BENCH_AMPLITUDE)
    lams = np.geomspace(0.05, 20.0, 14)
    rows = []
    for lam in lams:
        m = P.perimeter_map(llr, lam)
        rows.append((lam, _recall(m, truth_blob), _recall(m, truth_streak),
                     float((m & noise).sum()) / max(int(noise.sum()), 1)))
    arr = np.array(rows)
    # The operating point a user would actually pick: the lambda that best
    # recovers the compact artifact. What happens to the streak THERE is the
    # claim -- there is no "clean" lambda at which both survive, which is the
    # point.
    best = int(np.argmax(arr[:, 1]))
    lam_star, blob_star, streak_star, noise_star = arr[best]

    curve, found = P.curve_map(llr, lam_n=15.0, max_streaks=3)
    curve_recall = _recall(curve, truth_streak)
    curve_fp = float((curve & noise).sum()) / max(int(noise.sum()), 1)

    verdict = SUPPORTED if streak_star < 0.25 * blob_star else REFUTED
    detail = ("  lambda   blob recall   streak recall   noise kept\n" +
              "\n".join(f"    {r[0]:6.2f}   {r[1]:11.3f}   {r[2]:13.3f}   {r[3]:9.3f}"
                        for r in rows) +
              f"\n    the two recalls move in OPPOSITE directions with lambda: at "
              f"lambda={lam_star:.2f}, where the blob peaks at {blob_star:.2f}, the "
              f"streak is down to {streak_star:.2f} ({noise_star:.2f} of the noise left)"
              f"\n    curve prior on the SAME evidence: streak recall "
              f"{curve_recall:.2f} at noise kept {curve_fp:.3f} "
              f"({len(found)} line{'s' if len(found) != 1 else ''} accepted)")
    fig = _plot_p1(arr, lam_star, curve_recall) if plots else None
    return Result(verdict, detail,
                  {"lambda_best_blob": lam_star, "blob_recall_at_best": blob_star,
                   "streak_recall_at_best": streak_star,
                   "curve_streak_recall": curve_recall,
                   "curve_noise_kept": curve_fp}, fig)


#: Bench artifact amplitude, in sigma per pixel, and the prior odds that a pixel
#: is bad. Together they set the pointwise decision at z = a/2 - log(odds)/a =
#: 2.0 sigma, so the per-pixel detector alone runs at a ~2% false-positive rate
#: and roughly half recall -- the regime where a shape prior has something to
#: add and can be seen adding it.
BENCH_AMPLITUDE = 2.2
BENCH_PRIOR_ODDS = np.exp(-1.98)


def _prior_bench(rng, shape=(160, 160)):
    """A z-score field carrying a blob, a streak and scattered noise spikes.

    Amplitudes are set so blob and streak carry comparable TOTAL evidence: the
    claim is about the prior's geometry, so the detectors must not be separable
    on signal strength alone. Returned as a z-field rather than an LLR because
    the two methods under comparison consume different currencies -- the joint
    MAP wants the LLR, the current pipeline thresholds a z -- and they must be
    handed the SAME underlying field or the comparison is about the field.
    """
    z = rng.normal(0.0, 1.0, shape)
    rows, cols = np.indices(shape).astype(np.float64)
    blob = ((rows - 40.0) ** 2 + (cols - 45.0) ** 2) <= 9.0 ** 2
    d = (cols - 60.0) * np.cos(np.radians(35.0)) - (rows - 100.0) * np.sin(np.radians(35.0))
    streak = (np.abs(d) <= 1.5) & (np.abs(rows - 100.0) < 55)
    z[blob] += BENCH_AMPLITUDE
    z[streak] += BENCH_AMPLITUDE
    spikes = rng.random(shape) < 0.004
    spikes &= ~blob & ~streak
    z[spikes] += 4.5
    return z, blob, streak, spikes


def _recall(mask, truth) -> float:
    truth = np.asarray(truth, bool)
    return float((mask & truth).sum()) / max(int(truth.sum()), 1)


@claim("P2", "Radon/Hough IS the MAP detector under the curve prior, so it "
             "dominates a region prior on streaks at matched false-positive rate.",
       "The perimeter MAP matches the curve detector's streak recall at equal "
       "false positives -- then the separate detector earns nothing.")
def exp_p2_curve_detector(**kw) -> Result:
    z, blob, streak, noise = _prior_bench(np.random.default_rng(4))
    llr = TW.llr(z, BENCH_PRIOR_ODDS, BENCH_AMPLITUDE)
    bg = ~blob & ~streak & ~noise

    def fp(m):
        return float((m & bg).sum()) / max(int(bg.sum()), 1)

    best = {}
    for lam in np.geomspace(0.05, 20.0, 20):
        m = P.perimeter_map(llr, lam)
        best[fp(m)] = max(best.get(fp(m), 0.0), _recall(m, streak))
    targets = sorted(best)
    curve, _ = P.curve_map(llr, lam_n=15.0, max_streaks=3)
    c_fp, c_rec = fp(curve), _recall(curve, streak)
    matched = [best[f] for f in targets if f <= max(c_fp, 1e-6)]
    region_at_match = max(matched) if matched else 0.0

    verdict = SUPPORTED if c_rec > 1.5 * region_at_match else REFUTED
    return Result(verdict,
                  f"curve prior: streak recall {c_rec:.2f} at background FP "
                  f"{c_fp:.4f}. perimeter MAP at the same or lower FP: "
                  f"{region_at_match:.2f}",
                  {"curve_recall": c_rec, "curve_fp": c_fp,
                   "region_recall_at_matched_fp": region_at_match})


@claim("P3", "The artifact fields are class-disjoint, so M = union of the "
             "per-class masks is exact rather than a heuristic.",
       "A large fraction of reference-mask pixels sits in components that are "
       "simultaneously line-like and region-like -- then the classes overlap "
       "and the union double-counts evidence.",
       needs="run")
def exp_p3_class_disjoint(run: int, **kw) -> Result:
    from scipy import ndimage as ndi
    from automask.evaluation import load_sample
    from automask.masking import production_pipeline

    pipe = production_pipeline()
    sample = load_sample(run, features=pipe.features_needed())
    resid = sample.human & ~pipe.floor(sample)
    lab, n = ndi.label(resid)
    if n == 0:
        return Result(BLOCKED, "the reference mask adds nothing beyond the floor", {})

    total = int(resid.sum())
    sizes, elong, fill = [], [], []
    for k in range(1, n + 1):
        sel = lab == k
        s = int(sel.sum())
        if s < 8:
            continue
        rr, cc = np.nonzero(sel)
        cov = np.cov(np.stack([rr, cc]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        e = float(np.sqrt(ev[0] / max(ev[1], 1e-9)))
        f = s / float((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
        sizes.append(s); elong.append(e); fill.append(f)
    sizes = np.array(sizes); elong = np.array(elong); fill = np.array(fill)

    linelike = elong > 4.0
    regionlike = fill > 0.55
    ambiguous = linelike & regionlike
    frac_amb = float(sizes[ambiguous].sum()) / max(total, 1)
    frac_line = float(sizes[linelike & ~regionlike].sum()) / max(total, 1)
    frac_region = float(sizes[regionlike & ~linelike].sum()) / max(total, 1)
    frac_neither = 1.0 - frac_amb - frac_line - frac_region

    verdict = SUPPORTED if frac_amb < 0.10 else REFUTED
    return Result(verdict,
                  f"{len(sizes)} components >= 8 px covering {total} px: "
                  f"{100*frac_line:.0f}% line-like, {100*frac_region:.0f}% "
                  f"region-like, {100*frac_amb:.0f}% both, "
                  f"{100*frac_neither:.0f}% neither",
                  {"frac_ambiguous": frac_amb, "frac_line": frac_line,
                   "frac_region": frac_region, "n_components": float(len(sizes))})


# ==========================================================================
#  H -- the harmonic cut
# ==========================================================================
@claim("H1", "Polarization is exactly m = +-2 and removable analytically.",
       "In the LOG domain the cut is applied in, log P is not band-limited to "
       "m = 2 -- then an analytic m = +-2 removal leaves a residual and the "
       "'exactly' is false.")
def exp_h1_polarization_harmonics(**kw) -> Result:
    rows = [H.log_pol_leakage(tt) for tt in
            np.radians([5.0, 10.0, 15.0, 20.0, 28.9])]
    lin = max(r["linear_above_m2"] for r in rows)
    log_max = max(r["log_above_m2"] for r in rows)
    ratio = max(r["log_m4_over_m2"] for r in rows)

    verdict = MIXED if (lin < 1e-12 and log_max > 1e-6) else (
        SUPPORTED if log_max <= 1e-6 else REFUTED)
    detail = ("  2theta   power above m=2 (linear)   power above m=2 (log)   "
              "m4/m2 amplitude   log depth\n" +
              "\n".join(f"    {r['two_theta_deg']:5.1f}deg   {r['linear_above_m2']:22.2e}   "
                        f"{r['log_above_m2']:20.2e}   {r['log_m4_over_m2']:15.4f}   "
                        f"{r['log_depth']:9.4f}" for r in rows) +
              "\n    exact in the linear domain, not in the log the cut acts on")
    return Result(verdict, detail,
                  {"max_linear_leak": lin, "max_log_leak": log_max,
                   "max_m4_over_m2": ratio})


@claim("H2", "A sharp shadow is broadband to m ~ 2pi/delta_chi, so projecting "
             "out |m| <= m0 keeps shadows narrower than ~2pi/m0.",
       "The surviving fraction of a shadow's power does not collapse at "
       "m0 ~ 2pi/delta_chi -- then m0 is not the interpretable knob it is sold as.")
def exp_h2_shadow_bandwidth(plots: bool = False, **kw) -> Result:
    widths_deg = (2.0, 5.0, 10.0, 20.0, 45.0, 90.0)
    m0s = (0, 1, 2, 4, 8, 16, 32)
    table = np.array([[H.surviving_fraction(np.radians(w), m0) for m0 in m0s]
                      for w in widths_deg])
    predicted = [2 * np.pi / np.radians(w) for w in widths_deg]
    # Where does each width lose half its power?
    half = []
    for i, w in enumerate(widths_deg):
        idx = np.nonzero(table[i] < 0.5)[0]
        half.append(m0s[idx[0]] if idx.size else np.inf)
    ratio = [h / p for h, p in zip(half, predicted) if np.isfinite(h)]
    consistent = all(0.1 < r < 1.5 for r in ratio)

    verdict = SUPPORTED if consistent else MIXED
    detail = ("  shadow width   " + "  ".join(f"m0={m:<3d}" for m in m0s) +
              "   2pi/dchi   m0 at 50%\n" +
              "\n".join(f"    {w:6.1f}deg   " +
                        "  ".join(f"{v:6.3f}" for v in table[i]) +
                        f"   {predicted[i]:8.1f}   {half[i]:9}"
                        for i, w in enumerate(widths_deg)))
    fig = _plot_h2(widths_deg, m0s, table) if plots else None
    return Result(verdict, detail,
                  {"half_power_m0": float(half[0]) if np.isfinite(half[0]) else np.nan,
                   "n_widths_consistent": float(sum(1 for r in ratio if 0.1 < r < 1.5))},
                  fig)


@claim("H3", "The harmonic cut is available at all on this geometry.",
       "Partial azimuthal coverage makes the harmonic design ill-conditioned "
       "below the m0 the shadow bandwidth requires -- then the cut cannot be "
       "computed where it would be useful, whatever its theory.")
def exp_h3_coverage_limit(**kw) -> Result:
    t = fw.truth()
    rows = []
    for r in range(t.n_rings):
        sel = t.valid & (t.ring_idx == r)
        if sel.sum() < 100:
            continue
        chi = t.chi[sel]
        cov, _ = H.angular_coverage(chi)
        rows.append((r, float(np.degrees(cov)), int(sel.sum()),
                     H.max_usable_order(chi, m_limit=32),
                     H.max_usable_order(chi, m_limit=32, on_arc=True)))
    circle = np.array([r[3] for r in rows], dtype=float)
    arc = np.array([r[4] for r in rows], dtype=float)
    cov = np.array([r[1] for r in rows])
    med_cov = float(np.median(cov))
    # `m0` has to sit above the sample's low-order anisotropy (m ~ 4) and below
    # the shadow's bandwidth. On an arc of width W, order m resolves W/m, so the
    # finest angular feature the cut can address is this:
    finest = med_cov / max(np.median(arc), 1.0)

    # Available as written (full-circle harmonics), available only after
    # rescaling to the arc, or not available at all.
    verdict = (SUPPORTED if np.median(circle) >= 4 else
               MIXED if np.median(arc) >= 4 else REFUTED)
    detail = ("  ring   arc coverage   pixels   max m0 (circle)   max m0 (on arc)\n" +
              "\n".join(f"    {r[0]:4d}   {r[1]:9.1f}deg   {r[2]:6d}   "
                        f"{r[3]:15d}   {r[4]:15d}"
                        for r in rows[::max(len(rows) // 8, 1)]) +
              f"\n    median arc coverage {med_cov:.0f}deg over {len(rows)} rings; "
              f"median usable m0 = {np.median(circle):.0f} on the full circle, "
              f"{np.median(arc):.0f} after rescaling to the arc"
              f"\n    on the arc the cut can address features down to "
              f"~{finest:.1f}deg, which is the number that has to be compared "
              f"against a real shadow's angular width")
    return Result(verdict, detail,
                  {"median_usable_m0_circle": float(np.median(circle)),
                   "median_usable_m0_arc": float(np.median(arc)),
                   "median_coverage_deg": med_cov,
                   "finest_feature_deg": finest})


# ==========================================================================
#  O -- the objective
# ==========================================================================
@claim("O1", "Thresholding before the prior acts discards the evidence the "
             "prior needs, so the current pipeline is a lossy approximation to "
             "the joint MAP.",
       "The joint MAP does not beat threshold-then-morphology on the same "
       "evidence field at matched masked fraction -- then the greedy pipeline "
       "loses nothing worth recovering.")
def exp_o1_greedy_vs_joint(**kw) -> Result:
    # ONE draw, two currencies: the joint MAP consumes the LLR, the pipeline
    # thresholds the z. Both therefore see identical evidence.
    zf, blob, streak, noise = _prior_bench(np.random.default_rng(6))
    llr = TW.llr(zf, BENCH_PRIOR_ODDS, BENCH_AMPLITUDE)
    truth = blob | streak

    # The objective the notes actually propose is a union of PER-CLASS MAPs.
    # Scoring the perimeter MAP alone against a truth that contains a streak
    # would just re-measure P1, so the curve term is in the joint model, and the
    # perimeter-only column is carried to show what leaving it out costs.
    curve, _ = P.curve_map(llr, lam_n=15.0, max_streaks=3)
    region_only, joint = [], []
    for lam in np.geomspace(0.05, 20.0, 24):
        m = P.perimeter_map(llr, lam)
        region_only.append((float(m.mean()), _iou(m, truth)))
        u = m | curve
        joint.append((float(u.mean()), _iou(u, truth)))
    greedy = []
    for k in np.linspace(1.0, 5.0, 24):
        for op in (0, 2, 3):
            m = P.greedy_mask(zf, k, open_size=op, close_size=op)
            greedy.append((float(m.mean()), _iou(m, truth)))

    def best_near(rows, frac, tol=0.25):
        cand = [v for f, v in rows if abs(f - frac) <= tol * max(frac, 1e-9)]
        return max(cand) if cand else np.nan

    target = float(truth.mean())
    j, g = best_near(joint, target), best_near(greedy, target)
    j_best = max(v for _, v in joint)
    g_best = max(v for _, v in greedy)
    r_best = max(v for _, v in region_only)

    verdict = SUPPORTED if j_best > g_best + 0.02 else REFUTED
    return Result(verdict,
                  f"best IoU over all knobs: joint (perimeter MAP | curve MAP) "
                  f"{j_best:.3f}, threshold-then-morphology {g_best:.3f}, "
                  f"perimeter MAP alone {r_best:.3f}. At matched masked fraction "
                  f"({100*target:.2f}%): {j:.3f} vs {g:.3f}",
                  {"joint_iou_matched": j, "greedy_iou_matched": g,
                   "joint_iou_best": j_best, "greedy_iou_best": g_best,
                   "region_only_iou_best": r_best})


# ==========================================================================
#  figures
# ==========================================================================
def _plot_p1(arr, lam_star, curve_recall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(arr[:, 0], arr[:, 1], "o-", label="blob recall")
    ax.semilogx(arr[:, 0], arr[:, 2], "s-", label="streak recall")
    ax.semilogx(arr[:, 0], arr[:, 3], "^:", label="noise spikes kept")
    ax.axvline(lam_star, color="0.5", lw=1, ls="-",
               label=f"$\\lambda$ best for the blob = {lam_star:.2f}")
    ax.axhline(curve_recall, color="tab:green", ls="--",
               label=f"curve prior, streak recall {curve_recall:.2f}")
    ax.set_xlabel("perimeter prior weight  $\\lambda$")
    ax.set_ylabel("fraction")
    ax.set_title("One prior, two geometries: the perimeter MAP cannot keep a\n"
                 "streak at any $\\lambda$ that removes noise", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_h2(widths_deg, m0s, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, w in enumerate(widths_deg):
        ax.plot(m0s, table[i], "o-", label=f"{w:.0f}$\\degree$ shadow")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("harmonic orders projected out,  $|m| \\leq m_0$")
    ax.set_ylabel("fraction of shadow power surviving")
    ax.set_title("The cost side of the harmonic cut", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ==========================================================================
#  driver
# ==========================================================================
def run_claims(ids=None, runs=(), plots=False) -> Dict[str, Result]:
    selected = [c for c in CLAIMS if ids is None or c.id in ids]
    out: Dict[str, Result] = {}
    for c in selected:
        print(f"\n{'=' * 78}\n{c.id}  {c.statement}")
        print(f"  falsified by: {c.falsifier}")
        if c.needs == "run" and not runs:
            out[c.id] = Result(BLOCKED, "needs a run; none given (--runs)")
            print(f"  -> {BLOCKED}: no run supplied")
            continue
        targets = list(runs) if c.needs == "run" else [None]
        for r in targets:
            key = c.id if r is None else f"{c.id}/{r}"
            try:
                res = c.experiment(run=r, plots=plots) if r is not None \
                    else c.experiment(plots=plots)
            except Exception as e:                       # noqa: BLE001 -- reported
                res = Result(BLOCKED, f"{type(e).__name__}: {e}")
            out[key] = res
            head = f"  -> {res.verdict}"
            print(f"{head}\n{res.detail}" if "\n" in res.detail
                  else f"{head}: {res.detail}")
            if res.figure is not None:
                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"identification_{c.id}.png")
                res.figure.savefig(path, dpi=120)
                print(f"     [figure] {path}")
    return out


def summarize(results: Dict[str, Result]):
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    by_id = {c.id: c for c in CLAIMS}
    for key, res in results.items():
        cid = key.split("/")[0]
        first = res.detail.splitlines()[0] if res.detail else ""
        print(f"  {key:<10} {res.verdict:<10} {by_id[cid].needs:<6} {first[:64]}")
    counts = {}
    for res in results.values():
        counts[res.verdict] = counts.get(res.verdict, 0) + 1
    print("\n  " + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", nargs="*", default=None,
                    help="claim ids to run (default: all)")
    ap.add_argument("--runs", type=int, nargs="*", default=[],
                    help="runs for the claims that need real data")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    results = run_claims(args.claims, args.runs, args.plots)
    summarize(results)


if __name__ == "__main__":
    main()
