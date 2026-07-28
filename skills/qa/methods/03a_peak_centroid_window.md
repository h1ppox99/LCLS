---
name: xray-qa-peak-centroid-window
description: QA peak-position check, default implementation — local-baseline sub-bin centroid of the sample's main peak, verdict = centroid inside the manifest window. No functional-form assumption, so it degrades gracefully on asymmetric liquid peaks. Alternative implementation - 03b shape fit; exactly one produces the verdict per curve.
category: qa
role: peak-position (default implementation)
gate: exactly ONE of 03a/03b produces the verdict per curve (qa.peak_method); the other may run as a recorded cross-check
status: not-wired
---

# QA · 03a — Peak position: centroid vs window

Sub-bin centroid of the sample's main peak. Default implementation;
alternative: [03b_peak_shape_fit.md](03b_peak_shape_fit.md).

## Principle

What it checks:

Take the sample-peak candidate from feature attribution (the strongest
*sharp* feature inside `sample_features.<sample>.main_peak_q_window_A_inv`)
and refine it to sub-bin position:

```
support   = ± W bins around the maximum (W ≈ the feature's FWHM in bins)
baseline  = linear interpolation between the support edges (local, not global)
w         = I − baseline, clipped at 0
centroid  = Σ q·w / Σ w
σ_peak    = second moment of w about the centroid
u_centroid ≈ σ_peak / √N_eff,   N_eff = (Σw)² / Σw²
verdict   = centroid ∈ manifest window (uncertainty noted at the edges)
```

### Rationale and lineage

`argmax` alone is bin-quantized — at ~0.01 Å⁻¹ bins the quantization
error is the same order as the LaB6 drift tolerance, so a raw-argmax
verdict flips between neighboring bins on noise. This is the correction
to the legacy `qa_check.py` `find_peak` (pure argmax), the same way the
coarse-selection files correct the legacy percentile cut: keep the
question, fix the estimator.

The local baseline is not optional: the liquid main peak rides a
falling diffuse background, and an un-baselined centroid drags toward
low q by a systematic fraction of a bin — enough to matter exactly when
the verdict is contested, i.e. near a window edge.

The centroid's virtue is what it does NOT assume: no functional form,
so it degrades gracefully on the asymmetric peak shapes liquid
structure factors actually have.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.centroid_support_fwhm` | 1.0 | support half-width as a multiple of the feature FWHM |

The window itself is `sample_features.<sample>.main_peak_q_window_A_inv`
— never widened, never re-centered by this method.

## When to use

Default: one dominant, reasonably isolated peak with adequate
statistics — the normal case for a sample run.

### When NOT to use → [03b_peak_shape_fit.md](03b_peak_shape_fit.md)

- Attribution found a **second feature overlapping** the window: the
  centroid of a blend is the answer to no question.
- The verdict is **borderline** (centroid within ~1 u_centroid of a
  window edge): the shape fit's covariance-based uncertainty is
  better-founded; let it decide, record both.
- The local background is strongly **curved** across the support, so the
  linear edge interpolation is visibly wrong.

## Trade-offs

Failure modes / escalation:

- **hard** `main_peak_outside_window` — with the manifest's geometry,
  the peak is where it is. Do not tune L, beam center, or λ; do not
  re-window. Escalate with the zoom plot attached.
- **hard** `main_peak_absent` — no sharp candidate above the detection
  floor anywhere in the window: no sample in the beam, wrong sample
  name passed by the orchestrator, or normalization destroyed contrast.
  All three are stop-the-line facts.
- **soft** `main_peak_borderline` — centroid within its own uncertainty
  of a window edge. Requires the zoom plot; if `qa.peak_method`
  configures a cross-check, run it before recording the verdict.

## Outputs

Contributes to `qa_report.json`, per curve: `peak_q`, `peak_q_unc`, `peak_sigma`,
`peak_method: "centroid_window"`, `peak_verdict`, `baseline_slope`.

## Links

Part of: [qa](../README.md). Takes the sample-peak candidate from
[02 attribution](02_attr_feature_width.md). Alternative implementation:
[03b](03b_peak_shape_fit.md). Its estimator is also the sub-bin refiner for the
drift check ([06](06_calib_lab6_drift.md)) — one estimator per run.
