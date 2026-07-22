# Peak position: centroid vs window — sub-bin centroid of the sample's main peak

**Kind**: peak-position check (default implementation; alternative:
`peak_shape_fit.md` — exactly ONE implementation produces the verdict
per curve; the other may run as a recorded cross-check)

## What it checks

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

## Rationale and lineage

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

## When to prefer

Default: one dominant, reasonably isolated peak with adequate
statistics — the normal case for a sample run.

## When NOT to use → `peak_shape_fit.md`

- Attribution found a **second feature overlapping** the window: the
  centroid of a blend is the answer to no question.
- The verdict is **borderline** (centroid within ~1 u_centroid of a
  window edge): the shape fit's covariance-based uncertainty is
  better-founded; let it decide, record both.
- The local background is strongly **curved** across the support, so the
  linear edge interpolation is visibly wrong.

## Failure modes / escalation

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

## Contributes to `qa_report.json`

Per curve: `peak_q`, `peak_q_unc`, `peak_sigma`, `peak_method: "centroid_window"`,
`peak_verdict`, `baseline_slope`.
