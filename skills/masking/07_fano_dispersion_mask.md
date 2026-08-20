---
name: xray-masking-fano-dispersion
description: Across-shot Fano-factor (variance/mean) dispersion mask — an AGENT-DECIDED statistical mask layer. Cheap two-moment screen per pixel against the Poisson fingerprint F = 1, fenced by an MC-quantile null with the measured flux sequence. The only method that catches STUCK/saturated pixels (F < 0.5; their deviance is ~0). Direction is diagnostic: F >> 1 hot/flicker/noise, F < 1 over-processing/saturation. Run0475 (T=2764): 131/839,539 masked (0.016%, chance floor ~84), 0 stuck, 2 beam-vicinity pixels routed not masked.
category: masking
role: agent-decided statistical mask layer (across-shot dispersion)
gate: conditional — needs per-shot calibrated frames (qa photon-stats maps) and T*lambda >= min_photons per pixel; run after the signal-independent baseline layers
status: not-wired
---

# Masking · 07 — Fano-factor dispersion mask (agent decides)

Mask-side consumer of the across-shot statistics that [qa/08 pixel photon
statistics](../qa/methods/08_pixel_photon_statistics.md) computes (QA judges and
routes; this layer *acts*). Statistical family — judges pixels by their
*time-series dispersion*, not by geometry (00/01/03/04) or by the diffraction
pattern (02/05/06). Companion: [08_js_shape_mask.md](08_js_shape_mask.md) reads
the *shape* of the same histograms; this method reads only two moments.

## The decision the agent owns

> **Run this layer or not? At which false-positive target `mc_alpha`? And what
> to do with each class — mask, or route?**
>
> Needs a frames pass (~100 s for 2 764 shots). Classes are not all mask-worthy:
> beam-vicinity overdispersion is beam physics, not a pixel defect — masking it
> throws away the highest-signal pixels for no gain.

## Principle

For a healthy pixel at fixed flux, counts are Poisson, so **F = Var/Mean = 1**
exactly — the cheapest testable fingerprint (two streaming moments, no
histogram). Deviations are *directional*:

| Fano | Points at |
|---|---|
| F ≫ 1 | hot/unstable pixel, gain flicker, excess noise, correlated arrival (e.g. photon pairs: F = m for m-photon bunches — a 2nd-harmonic pixel sits at F ≈ 1.3) |
| F ≈ 1 + λ·CV² | *expected* flux-jitter overdispersion — absorbed by the measured-f_t null, never flagged |
| F < 1 mildly (0.95–0.98) | charge sharing + photonization rounding — known-benign, never flagged |
| F < 0.5 | stuck / saturating / over-processed — **deviance is ≈ 0 here; only Fano catches this class** |

The fence is the **MC-quantile null**: simulate `Poisson(λ·f_t)` with the
measured per-shot monitor sequence on a log-λ grid, take the empirical
`1 − mc_alpha` quantile of F, interpolate in log λ. No moment-based (MAD)
fences and no absolute thresholds — see Do not.

## Parameters

| CLI flag | Default | Meaning |
|---|---|---|
| `--min-photons` | 30 | fence only pixels with T·λ̂ ≥ this (statistic gate, see Do not) |
| `--mc-alpha` | 1e-4 | per-pixel false-positive target; expected chance flags = α·n_fenceable |
| `--mc-reps` | 20000 | MC replicas per λ-grid point (must resolve the 1−α quantile) |
| `--stuck-fano` | 0.5 | absolute lower backstop for the stuck class (validated: 100 % detection, 0 FP in qa/08's self-test) |
| `--beam-radius-px` | 60 | assembled radius inside which overdispersion is classed `beam_modulated` and routed, not masked |
| `--monitor-band`, `--max-shots` | must match the photon-stats run | the f_t sequence of the null |

## Decision rules

- **RUN** when frames are available and the product benefits from a
  statistically-vetted pixel set (calibration sums, weighted averaging), or when
  qa/08 routed flags here. **SKIP** (recorded) when no frames, or when
  T·λ leaves too few fenceable pixels to matter (report the gate count).
- **Use the full screened shot set** (wide band + measured-f_t null), not a
  narrow monitor band: power scales with T·λ, and the conditional null absorbs
  even CV ≈ 1 jitter (validated).
- Routing per class: `overdispersed` → mask layer; `stuck` → mask layer;
  `beam_modulated` (over fence but inside `beam_radius_px`) → **not masked**,
  reported to normalization/QA as pointing-sensitivity evidence; F just *under*
  the fence with suspicious signature → hand to [08](08_js_shape_mask.md),
  which has more power for shape-type anomalies.

## Implementation

`scripts/statistical_pixel_masks.py --stat fano`, consuming
`photon_stats.npz` from `../qa/scripts/photon_stats.py` (run that first, wide
band). Writes `mask_fano.npy` (bool layer, True = exclude) + report JSON.

## Evidence (Run0475, 2026-08-04)

T = 2 764 shots (13–100 % band, monitor CV 1.04), 839 539 fenceable pixels
(184 k below the T·λ gate):

- **131 masked (0.016 %)** at α = 1e-4 — chance floor ≈ 84, so ≲ 50 real
  overdispersed pixels; **0 stuck** (clean detector); **2 beam-vicinity pixels
  routed** not masked. Global Fano median 0.992.
- The known 2nd-harmonic pixel (0,419,1022) sits at F = 1.301 with its fence at
  ≈ 1.31 — *missed by a hair*: Fano is the screen, not the classifier. Its JS
  is 6× its own null p95 → caught by [08](08_js_shape_mask.md). Six pixels land
  in BOTH masks, all with the same signature (F ≈ 1.3, JS ≈ 5e-4) — a candidate
  harmonic-ring population, routed to beamline diagnostics.
- **Negative result that shaped the method**: median + k·MAD fences (qa/08's
  convention for deviance) mass-fail on Fano — at T·λ ≈ 2–8 the within-stratum
  MAD is *exactly zero* (counts on a lattice) and the fence degenerates to the
  median: 7 702 false masks (0.87 %) on the 317-shot band, 166 k (20 %!) at
  T = 2 764. The MC quantile reproduces the lattice and fixes this.

## When to use

Any run with per-shot frames where a cheap, directional, statistically
calibrated pixel screen is wanted — especially to catch stuck/saturated pixels
that every deviance/shape test is blind to.

## Trade-offs

- Two moments only: blind to shape (bimodal flicker at modest variance,
  pair-arrival just under the fence). Pair with 08 when shape matters.
- Power gate excludes dim pixels (T·λ < 30) — on a short run that can be most
  of the detector; the gate count must be reported, not hidden.
- `mc_alpha` trades false masks against sensitivity; at 1e-4 the chance floor
  (~84 px) is of the same order as the real population on a healthy detector.

## Do not

- **Don't fence Fano with median + k·MAD** — lattice collapse, measured above.
- **Don't fence on absolute F = 1** with a jittering beam (qa/08 rule) — the
  measured-f_t MC null is the reference.
- **Don't fence below the T·λ gate** — report `n_below_gate` instead.
- **Don't mask the `beam_modulated` class** — route it; it is beam physics.
- Don't recompute λ̂ from a different shot set than the f_t null.

## Outputs

`mask_fano.npy` (bool, `(2,512,1024)`, True = exclude — one disjoint layer for
the mask union, logged with count and %); `mask_fano_report.json`: gate counts,
fences (`mc` grid + quantiles), per-λ-stratum diagnostics (median/MAD, reported
only), class counts, mask fraction.

## Machine block

```json
{
  "method": "fano_dispersion_mask",
  "defaults": {
    "min_photons": 30.0,
    "mc_alpha": 1e-4,
    "mc_reps": 20000,
    "stuck_fano": 0.5,
    "beam_radius_px": 60.0
  },
  "run0475": {
    "t_shots": 2764,
    "n_fenceable": 839539,
    "n_mask": 131,
    "mask_fraction": 0.00016,
    "stuck_masked": 0,
    "beam_modulated_routed": 2,
    "global_fano_median": 0.992
  }
}
```

## Links

Part of: [masking](README.md). Companion: [08_js_shape_mask.md](08_js_shape_mask.md)
(shape classifier for what this screen suspects but cannot resolve). Consumes:
[qa/08](../qa/methods/08_pixel_photon_statistics.md) maps (`photon_stats.npz`).
Related evidence: `outputs/statistical_masks_20260804/`.
