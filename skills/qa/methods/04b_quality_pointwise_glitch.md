---
name: xray-qa-quality-pointwise-glitch
description: QA curve-quality check, localized lens — single-bin spikes/notches/steps via rolling-median residuals in robust noise-sigma units, each event type attributed to a different upstream cause (mask escape, over-masking, panel-boundary stitching). MUST run after 02 attribution so labeled physics is excluded from fencing. Composes with the global smoothness lens 04a.
category: qa
role: curve-quality (localized lens)
gate: run what qa.quality_checks lists; MUST run after 02 attribution — labeled features are excluded from fencing
status: not-wired
---

# QA · 04b — Curve quality: pointwise glitches

Localized artifacts via rolling-median residuals. Composes with the smoothness lens
[04a_quality_smoothness_curvature.md](04a_quality_smoothness_curvature.md).

## Principle

What it flags:

```
baseline = rolling median of I, width w = qa.glitch_window_bins (default 11)
resid    = (I − baseline) / σ_noise          (robust σ from MAD(ΔI)/√2, as in attribution)
glitch   = |resid| > qa.glitch_k (default 6), outside all attributed-feature windows
cluster into events:
    spike  — 1–2 bins, positive
    notch  — 1–2 bins, negative
    step   — persistent baseline shift
```

### Why localized evidence needs its own lens

The global L2 dilutes a single-bin artifact below visibility. More
importantly, each event *type* attributes to a different upstream cause
— the value of this check is the routing, not the count:

| Event | Points at |
|---|---|
| spike | hot pixel(s) escaped the mask into that q-bin. Cross-reference the mask stage: the ring-statistics refinement should have caught it; `refinement_added_fraction` sitting at its cap suggests it ran out of budget |
| notch | over-masking concentrated in one ring — the shape the "no azimuthal arcs" rule exists to prevent; check `mask_layers.npz` for which layer killed that ring's pixels |
| step  | correction-map or stitching error where a panel boundary crosses the q-rings; evidence for the mask skill's geometry escalations, not a reason to mask more pixels |

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.glitch_window_bins` | 11 | rolling-median width (odd) |
| `qa.glitch_k` | 6 | fence in robust noise-σ units — loose; ~10³ bins tested per curve, same multiple-testing logic as the mask refinements |

## Decision rules

### Ordering dependence (hard requirement)

Runs AFTER feature attribution. With w = 11 bins (~0.1 Å⁻¹ at the
drift-check binning), a real σ ≈ 0.01 Å⁻¹ Bragg ring IS a glitch to
this filter — on a calibration run, unexcluded rings would fill the
report with physics. Exclude every labeled feature's support
(± 2 σ_feat); if a fence fires *at the boundary* of an exclusion zone,
widen the zone, don't lower k.

## Trade-offs

Failure modes / escalation:

- **soft** `curve_glitch_detected` — per event: q, type, amplitude in
  σ-units, suspected route from the table above.
- No hard escalation of its own: a glitch density that renders the
  curve unusable (> ~5 % of bins fenced) reroutes to the integrity
  gate's `curve_integrity_failed` instead.
- **Quantile noise on sparse tails**: at extreme q where bins are
  thinly populated, the rolling median itself jitters — exclude bins
  the integrity gate already marked starved before fencing.

## Outputs

Contributes to `qa_report.json`, per curve: `glitch_events` list —
`{q, type, amplitude_sigma, suspected_cause}`, `glitch_bin_fraction`.

## Links

Part of: [qa](../README.md). Global sibling lens:
[04a](04a_quality_smoothness_curvature.md). Feature exclusion zones come from
[02 attribution](02_attr_feature_width.md); unusable glitch density (> ~5 % of bins)
reroutes to the [01 integrity gate](01_gate_curve_integrity.md).
