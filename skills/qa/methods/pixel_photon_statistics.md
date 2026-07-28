# Pixel photon statistics — per-pixel Poisson goodness-of-fit across shots

**Kind**: detector-statistics check, pixel-level (gated on per-shot
calibrated frames being available — this is the only QA method that
reads evidence *upstream* of the 1D curves; it judges the detector,
not the integration)

## What it flags

Pixels whose shot-to-shot intensity fluctuations are inconsistent with
ideal Poisson photon statistics: hot pixels, gain instability
(two-state flicker), excess readout/common-mode noise, stuck or
saturating pixels. Also, as a *global* verdict: whole-detector over- or
underdispersion that points at selection/normalization rather than at
individual pixels.

## The sample definition (load-bearing)

The statistical sample is **one fixed pixel observed across T shots**:

```
I_i(1), I_i(2), ..., I_i(T)      →      λ̂_i = mean_t I_i(t)
baseline: I_i(t) ~ Poisson(λ̂_i)
```

**Never pool all pixels of one image into a single histogram and
compare it to one Poisson.** Different pixels have different expected
intensity (q-dependence, beam-center distance, module gain, shadows,
polarization), so the pooled distribution is a Poisson *mixture*
Σᵢ wᵢ·Poisson(k; λᵢ), which is not Poisson even when every pixel is
perfectly healthy — the pooled divergence measures the intensity
*landscape*, not detector health. (The self-test demonstrates this
numerically: pooled-KL on fully healthy synthetic data is orders of
magnitude above any per-pixel score.) Single-image, same-λ variants
(one narrow q-bin, a flat-field region) are legitimate but are a
different check; this method is the across-shots form.

## Photon conversion first

Raw Jungfrau output is ADU, not photon counts; the honest model is
Poisson–Gaussian: `Y = g·N + b + ε`, `N ~ Poisson(λ)`,
`ε ~ N(0, σ_read²)`. Comparing raw ADU to pure Poisson measures gain,
pedestal, and readout noise — not anomalous pixel behavior. Two valid
routes:

1. **(implemented)** pedestal-subtract, gain-correct, photonize:
   `k = max(0, round(keV / photon_kev))`. At 9.6 keV/photon and
   σ_read ≲ 0.5 keV the rounding is reliable, so the pure-Poisson
   baseline applies.
2. (alternative, for low photon energy or high read noise) keep ADU
   and fit the Poisson–Gaussian mixture as the baseline.

Known confounders after photonization — expected, not anomalies:

- **Flux jitter**: λ_i(t) = λ_i · f(t) with per-shot flux f(t) makes
  every pixel overdispersed by F ≈ 1 + λ·CV², where CV is the
  coefficient of variation of f. Mitigate by selecting shots inside a
  narrow monitor band (`monitor_band`), and judge against per-λ-stratum
  fences (below), never against an absolute F = 1.
- **Charge sharing**: photons split across pixel boundaries land as
  fractional keV and round inconsistently — a mild, spatially smooth
  distortion, strongest on ring gradients.
- **Threshold noise**: pixels whose keV values sit near k+0.5
  boundaries pick up rounding variance.

## Estimator hierarchy (why deviance is primary, KL is not)

With T ~ a few hundred shots the empirical PMF of one pixel is sparse;
a histogram-based KL estimate is dominated by sampling noise (many
bins, few counts, and P̂(k) > 0 where Q(k) ≈ 0 diverges). Ordered by
robustness at small T:

| Statistic | Formula (per pixel) | Role |
|---|---|---|
| Poisson deviance | `D = 2 Σ_t [I(t)·ln(I(t)/λ̂) − (I(t) − λ̂)]`; with MLE λ̂ this reduces to `2[Σ I ln I − S·ln(S/T)]`, S = Σ I | **primary verdict** — no PMF estimation needed |
| Fano factor | `F = Var(I)/λ̂` | quick screen + direction: F ≫ 1 overdispersion (instability, extra noise), F < 1 underdispersion (processing, saturation, correlations) |
| JS divergence | `JS(P̂ ‖ Poisson(λ̂))`, tail bins k ≥ K merged, ε-smoothed `P̂_ε(k) = (n_k + ε)/(T + εK)` | supplementary *shape* feature — symmetric and bounded, preferred over raw KL; separates flicker (bimodal) from plain excess noise at equal deviance |

One estimator set per run — mixing estimators across runs mixes their
systematics into cross-run comparisons (same rule as the peak-position
methods).

## Verdict: two fences, pixelwise max

The null distribution of deviance/dof depends on λ (χ² only
asymptotically; badly wrong for λ ≲ 1 — on Run0475 the null median
runs 0.02–0.13 across the λ range), and residual flux jitter shifts
it further. Two complementary fences, flag on
`dev/dof > max(fence_stratum, fence_MC)`:

1. **Empirical stratum fence** — bin pixels into *equal-count* λ̂
   strata (quantile edges; log-spaced strata leave thin
   contamination-dominated top strata that anomalies migrate into by
   inflating their own λ̂), take median + fence_k·MAD of dev/dof
   within each stratum, excluding known-bad pixels. The pixel
   population itself is the null — the same multiple-testing logic as
   the mask refinements. Blind spot: the null level drifts *within* a
   stratum, so pixels at a stratum's bright edge fence spuriously
   (before the MC fence this flagged 100 % of λ > 0.1 pixels on
   Run0475).
2. **MC-calibrated conditional null fence** — simulate
   `Poisson(λ·f_t)` on a log-λ grid using the **measured per-shot
   monitor sequence** f_t (the conditional-model idea applied to the
   fence itself), take median + fence_k·MAD of the simulated dev/dof,
   interpolate in log λ. Gives every pixel a null at its own λ with
   the actual flux jitter built in — even a CV = 0.5 band is absorbed
   correctly. Blind spot: models only flux jitter, not e.g.
   population-wide calibration drift — which the stratum fence
   catches.

Absolute Fano and the predicted `1 + λ·CV²_monitor` curve are
*reported* alongside for physics interpretation, but the flag comes
from the combined fence.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.photon_stats.photon_kev` | — (required) | single-photon energy for photonization |
| `qa.photon_stats.min_shots` | 200 | below this, skip loudly (`insufficient_shots`) — sparse-T deviance fences are unstable |
| `qa.photon_stats.monitor_band` | [0.25, 0.75] | keep shots between these monitor quantiles (flux-jitter control) |
| `qa.photon_stats.fence_k` | 6 | deviance fence in stratum-MAD units |
| `qa.photon_stats.lambda_strata` | 16 | log-spaced λ̂ strata for fencing |
| `qa.photon_stats.js_kmax` | 12 | histogram cap; k ≥ K merged into tail bin |
| `qa.photon_stats.js_epsilon` | 0.5 | additive smoothing count |
| `qa.photon_stats.max_flag_fraction` | 0.005 | hard escalation if exceeded (see below) |

## Failure modes / escalation

- **soft** `pixel_photon_stats_flagged` — per flagged pixel (or
  cluster): panel, (row, col), λ̂, dev/dof, Fano, JS, suspected class.
  Routing by signature:

  | Signature | Points at |
  |---|---|
  | isolated pixel, F ≫ 1, huge deviance | hot/unstable pixel escaped the mask → mask stage |
  | bimodal PMF (high JS at moderate deviance), F > 1 | gain-mode flicker or unstable calibration constant → calib |
  | spatially clustered flags on a panel edge/ASIC | common-mode or geometry correction → reduction/calib escalation |
  | global median F ≫ 1 + λ·CV² prediction | flux normalization or selection too loose → reduction |
  | global F < 1 | over-processing (double common-mode subtraction), saturation → reduction |

- **hard** `pixel_photon_stats_failed` — flagged fraction (outside the
  known-bad map) > `max_flag_fraction`: the detector state disagrees
  with the calibration wholesale; do not trust curves from this run.
- **skip** `insufficient_shots` / `frames_unavailable` — recorded with
  reason; never silent.

## Do not

- Don't pool pixels into one histogram (mixture pitfall above).
- Don't fence on absolute Fano = 1 with a jittering beam.
- Don't mask pixels from inside this skill — QA judges; the flag list
  is evidence routed to the mask/calib stages.
- Don't compare raw-ADU distributions to pure Poisson.

## Validated feasibility (2026-07-27, Run0475)

Implementation: `scripts/photon_stats.py` (self-test: `--selftest`).

- **Synthetic self-test** (160 k pixels × 600 shots, λ log-uniform
  0.01–20, 300 injected pixels per class): hot 100 % detected at
  λ ≥ 1 (96 % at 0.1–1), flicker 94 % at λ ≥ 1, anomalous-noise
  (σ = 2 photons) 100 % below λ = 1, stuck 100 % via Fano < 0.5;
  false-positive rate 0 at fence_k = 6; good-pixel dev/dof median
  0.94–0.97, Fano median 0.99–1.01. Identical performance with 10 %
  flux jitter. Pooled-histogram control: KL of pooled *healthy*
  pixels vs single Poisson = 1.54, ~3400× the per-pixel JS median —
  the mixture pitfall, demonstrated.
- **Real data** (Run0475, 317 shots in the ipm2 45–55 % band,
  CV = 0.10, ~19 s for 600-shot calibration+accumulation on a laptop):
  0 of 880 k fenceable pixels flagged — detector consistent with
  Poisson(λ·f_t) outside the known-bad map; global Fano median 0.997.
  Same result on the 25–75 % band (CV = 0.50): the measured-f_t MC
  null absorbs even that jitter. Median occupancy is ~0.006
  photons/pixel/shot, so per-λ nulls are far from χ² (medians
  0.02–0.13) — absolute thresholds would be wholesale wrong.
- **Injection into real frames** (300 healthy pixels per class,
  λ ∈ 0.001–0.1): hot 100 %, noise 100 %, stuck 100 %, off-target
  flags 0/880 k. Flicker 0 % — correctly invisible: ×3 two-state gain
  at λ ≤ 0.1 with T = 317 has no statistical power (see sample-size
  note above); flicker detection needs λ ≳ 1 or many more shots.
- Mild real effects seen and correctly *not* flagged: slight
  underdispersion (Fano ≈ 0.95) at mid-λ (charge sharing + rounding),
  and Fano ≈ 1.2–1.45 above the flux-jitter prediction on the few
  dozen brightest pixels (beam-vicinity pointing jitter).

## Contributes to `qa_report.json`

`photon_stats`: `{n_shots_used, monitor_cv, n_flagged, flag_fraction,
overlap_with_known_bad, global_fano_median, strata: [{lambda_range,
n_pixels, dev_dof_median, dev_dof_mad, fano_median, n_flagged}],
flags: [{panel, row, col, lambda, dev_dof, fano, js, suspected_class}],
maps: "photon_stats.npz"}`.
