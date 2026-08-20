---
name: xray-masking-js-shape
description: Across-shot JS-divergence shape mask — an AGENT-DECIDED statistical mask layer. Compares each pixel's empirical count histogram (12 head bins + merged tail, eps-smoothed) to Poisson(lambda_hat) with the symmetric, bounded Jensen-Shannon divergence, fenced by an MC-quantile null with the measured flux sequence. Catches SHAPE anomalies that pass a variance screen - bimodal gain flicker, pair-arrival / 2nd-harmonic contamination (deficit of singles + excess of doubles). Run0475 (T=2764): 66/839,539 masked (0.008%, ~chance floor); caught the harmonic pixel that the Fano fence missed by a hair.
category: masking
role: agent-decided statistical mask layer (across-shot histogram shape)
gate: conditional — needs per-shot calibrated frames (qa photon-stats maps), T*lambda >= min_photons, and enough shots for histogram power; run after 07 or on its suspects
status: not-wired
---

# Masking · 08 — JS-divergence shape mask (agent decides)

The shape-sensitive companion of [07_fano_dispersion_mask.md](07_fano_dispersion_mask.md):
07 reads two moments of the across-shot histogram, this method reads the whole
(binned) histogram. Same conditional null, same MC-quantile fencing, same
consumer relationship to [qa/08](../qa/methods/08_pixel_photon_statistics.md) —
different failure modes covered.

## The decision the agent owns

> **Run on all fenceable pixels, or only on 07's suspects? At which `mc_alpha`?
> And is a shape flag a mask, or evidence for another subsystem?**
>
> A pair-arrival signature (excess k=2) is usually *spectral contamination*
> (beamline 2nd harmonic), not a broken pixel — masking is defensible for
> clean-sum products, but the finding itself belongs to beamline diagnostics.

## Principle

Two distributions can share mean and variance and still differ in shape. Per
pixel, compare the empirical PMF `P̂(k)` across shots (bins 0..11 + merged tail
`k ≥ 12`, additive `ε = 0.5` smoothing — qa/08's exact convention) with
`Poisson(λ̂)` via

`JS = ½·KL(P̂‖M) + ½·KL(Q‖M),  M = ½(P̂+Q)`

— symmetric and bounded (no divergence on empty bins), preferred over raw KL
(qa/08 estimator hierarchy). What shape adds over dispersion:

| Signature | Fano sees | JS sees | Points at |
|---|---|---|---|
| bimodal PMF (two λ states) | F > 1, may sit under fence | high | gain flicker / unstable calib → calib |
| deficit of k=1 + excess of k=2 | F ≈ 1.3, borderline | high | pair arrival: 2nd-harmonic photons or grain twinkle → beamline |
| plain variance inflation | high | moderate | noise → mask |

Fence: **MC-quantile null** — `Poisson(λ·f_t)` replicas with the measured
monitor sequence on a log-λ grid, empirical `1 − mc_alpha` quantile of JS,
interpolated in log λ. Same lattice-safety argument as 07: at low T·λ the JS
values are quantized ("one k=2 event" is a lattice step), MAD-based fences
degenerate, quantiles of the simulated lattice do not.

## Parameters

| CLI flag | Default | Meaning |
|---|---|---|
| `--min-photons` | 30 | T·λ̂ gate (histogram power; below it JS is a few-lattice-point statistic) |
| `--mc-alpha` | 1e-4 | per-pixel false-positive target |
| `--mc-reps` | 20000 | MC replicas per grid point |
| `--beam-radius-px` | 60 | `beam_modulated` routing radius (as in 07) |
| `js_kmax = 12`, `js_epsilon = 0.5` | fixed | inherited from the qa/08 map computation — must match |

## Decision rules

- **RUN after 07** as its classifier (07's suspects + near-fence pixels), or
  standalone when hunting shape-only anomalies (spectral contamination, flicker).
  **SKIP** (recorded) when T·λ power is absent — a narrow-band run has none:
  the harmonic pixel's expected pair count in 317 shots is 0.57, invisible; at
  T = 2 764 it is ~5, decisive. Use the full screened shot set.
- Flags are classified before masking: `beam_modulated` routed (as in 07);
  pair-signature pixels masked for clean-sum products **and** reported to
  beamline diagnostics with their (λ, F, JS) triplet — if several align on a
  ring radius, that is a harmonic diffraction ring, a beam property.

## Implementation

`scripts/statistical_pixel_masks.py --stat js`, consuming `photon_stats.npz`
(run `../qa/scripts/photon_stats.py` first on the wide band). Writes
`mask_js.npy` + report JSON.

## Evidence (Run0475, 2026-08-04)

Same run as 07 (T = 2 764, 839 539 fenceable):

- **66 masked (0.008 %)** at α = 1e-4 — at/below the ≈ 84 chance floor: the
  detector has essentially no shape-anomalous population. Global JS median
  1.1e-5.
- **The division-of-labor exhibit**: pixel (0,419,1022) — the validated
  2nd-harmonic absorber (5 pair events, all 17.6–19.3 keV; singles exactly
  Poisson after removing them) — has F = 1.301, a hair *under* its Fano fence,
  but JS = 5.4e-4 ≈ 6× its own null p95: **caught here, missed by 07**. Shape
  keeps the information that the second moment compresses away.
- 6 pixels flagged by BOTH layers, every one with the same harmonic signature
  (F ≈ 1.3, JS ≈ 5e-4, λ ≈ 0.014) — candidate harmonic-ring population.
- Inherits 07's negative result: MAD fences mass-fail on lattice statistics
  (0.87–20 % false masks before the MC-quantile fence); JS additionally showed
  in the 100-pixel MC study (2026-08-03) that raw-value ranking without a
  per-λ null misreads dim pixels as "perfect" (KL ≈ 0 by data starvation).

## When to use

When 07 (or qa/08) leaves suspects unresolved, when spectral contamination is
suspected, or when a run's products are sensitive to non-Poisson pixel shape
(photon-counting analyses, speckle statistics).

## Trade-offs

- Costlier than 07 (full histograms; vectorized MC still dominates runtime).
- Power needs T·λ: both the gate and the shot count matter — this is the
  method where "merge more shots" pays off most directly.
- At α = 1e-4 the chance floor can exceed the real population on a healthy
  detector — report counts against `alpha·n_fenceable`, never bare counts.

## Do not

- **Don't fence JS with median + k·MAD** (lattice collapse — 07's Evidence).
- **Don't rank pixels by raw JS across different λ** — a dim pixel's tiny JS is
  data starvation, not health; only the per-λ null makes values comparable.
- **Don't run on a narrow monitor band for shape hunting** — no power (0.57
  expected pair events vs 5; measured).
- **Don't mask the `beam_modulated` class**; and don't silently mask
  pair-signature pixels without reporting the spectral finding.
- Don't change `js_kmax`/`js_epsilon` between the map computation and the null.

## Outputs

`mask_js.npy` (bool layer, True = exclude, disjoint-layer accounting);
`mask_js_report.json`: gate counts, MC grid + quantile fences, stratum
diagnostics (reported only), class counts, mask fraction.

## Machine block

```json
{
  "method": "js_shape_mask",
  "defaults": {
    "min_photons": 30.0,
    "mc_alpha": 1e-4,
    "mc_reps": 20000,
    "beam_radius_px": 60.0,
    "js_kmax": 12,
    "js_epsilon": 0.5
  },
  "run0475": {
    "t_shots": 2764,
    "n_fenceable": 839539,
    "n_mask": 66,
    "mask_fraction": 8e-05,
    "beam_modulated_routed": 1,
    "overlap_with_fano_mask": 6,
    "harmonic_pixel_caught": true
  }
}
```

## Links

Part of: [masking](README.md). Companion: [07_fano_dispersion_mask.md](07_fano_dispersion_mask.md)
(cheap screen; run first). Consumes: [qa/08](../qa/methods/08_pixel_photon_statistics.md)
maps. Related evidence: `outputs/statistical_masks_20260804/`,
`outputs/pixel_kl_map_20260803/` (100-pixel MC study, full-detector KL maps).
