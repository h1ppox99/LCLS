---
name: xray-qa-attr-feature-width
description: QA feature attribution — detect every local maximum and label it by width (sharp/diffuse/ambiguous) as sample peak, calibration ring, named instrument background, or PARASITIC candidate. Always runs second; produces labels, never a verdict; every downstream check consumes them.
category: qa
role: attribution
gate: always — runs SECOND, after the integrity gate; downstream checks consume its labels
status: not-wired
---

# QA · 02 — Attribution: feature width

Label every detected feature before any verdict logic. Produces labels,
never a pass/fail verdict of its own.

## Principle

What it does:

Detect all local maxima above the noise floor across the full q-range
(prominence-based, not height-based — a small peak on a low background
is still a feature), estimate each feature's width, then classify:

```
σ_feat ≤ qa.bragg_sigma_max_A_inv   (default 0.02)  → sharp
σ_feat ≥ qa.diffuse_sigma_min_A_inv (default 0.04)  → diffuse
in between                                          → ambiguous (report, don't force)

sharp   & q in main_peak_q_window_A_inv        → sample-peak candidate (peak-position family takes it)
sharp   & q at a LaB6 prediction               → calibration ring (drift check takes it)
diffuse & q in an instrument_backgrounds entry → labeled background, by name (e.g. kapton)
sharp   & none of the above                    → PARASITIC candidate — the load-bearing class
```

### Rationale

Pass/fail logic on unlabeled features conflates three different
stories. Width is the discriminator this detector affords: Bragg rings
arrive at σ ≈ 0.01 Å⁻¹, known instrument backgrounds at σ ≈ 0.04–0.08 —
a ≥ 4× separation, robust to reasonable binning choices.

The `sharp & unattributed` class is why this file exists: on an
amorphous sample (liquid CO₂ here) every persistent sharp ring is by
definition parasitic — ice buildup, window crystallites. This is the 1D
counterpart of the mask skill's `refine_pyfai_separate` evidence. If
that refinement ran, cross-reference its saved `bragg_2d` component:
the same scatterer seen in 2D and 1D is confirmation; seen only in 1D
means it escaped the mask (e.g. it lives in a condition-subset curve
and diluted out of the run sum the mask was built from).

Labeled backgrounds double as coarse q-scale witnesses: the Kapton
feature appearing far outside its manifest window is calibration
evidence — hand it to the drift check, do not silently re-label it.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.bragg_sigma_max_A_inv` | 0.02 | upper width bound for "sharp" |
| `qa.diffuse_sigma_min_A_inv` | 0.04 | lower width bound for "diffuse" |
| `qa.feature_min_prominence_sigma` | 5 | detection prominence in noise-σ units (loose — ~10³ bins tested per curve) |

Noise σ per bin: curves carry no per-bin errors (`np.savez(q=, I=)`
per the common SKILL), so estimate robustly from first differences:
`σ ≈ 1.4826 · MAD(ΔI) / √2`. Record the estimate.

## Trade-offs

Failure modes / escalation:

- **soft** `unattributed_sharp_feature` — q, width, intensity in the
  report. On amorphous samples this is operational news (ice forming
  during the run), not necessarily a pipeline bug — say so.
- **soft** `feature_width_ambiguous` — width in the 0.02–0.04 gap;
  report both candidate labels rather than forcing one.
- **soft** `instrument_background_missing` — a manifest-listed
  background absent from the curve. May be legitimate (setup change);
  note it, and check whether it *moved* rather than vanished.

## Do not

- **Do not add a feature to `instrument_backgrounds` mid-run to exclude
  it from pass/fail.** Manifest edits chasing a verdict are the
  geometry-tuning anti-pattern in a different coat. Propose the edit
  with evidence from a clean calibration run; a human adopts it.

## Outputs

Contributes to `qa_report.json`, per curve: `features` list —
`{q, sigma, prominence, label, matched_manifest_entry}`.

## Links

Part of: [qa](../README.md). Labels feed the peak-position family
([03a](03a_peak_centroid_window.md)/[03b](03b_peak_shape_fit.md)), the glitch lens
([04b](04b_quality_pointwise_glitch.md)), control windows
([05](05_control_windows.md)), and the drift check ([06](06_calib_lab6_drift.md)).
