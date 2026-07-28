---
name: xray-qa-peak-shape-fit
description: QA peak-position check, alternative implementation — Gaussian + linear-background model fit over the manifest window, covariance-based uncertainty, two-component variant for overlapping candidates. Prefer for structured or borderline peaks; guarded against silent fit failures. Same verdict contract as 03a; exactly one produces the verdict per curve.
category: qa
role: peak-position (alternative implementation)
gate: per 03a's "When NOT to use" — overlapping features, borderline verdicts, curved background; exactly ONE of 03a/03b produces the verdict
status: not-wired
---

# QA · 03b — Peak position: shape fit

Model fit with defensible uncertainties for structured or borderline peaks.
Alternative implementation of the same contract as
[03a_peak_centroid_window.md](03a_peak_centroid_window.md).

## Principle

What it checks:

Same verdict contract, via a model fit over the manifest window:

```
I(q) = A · exp(−(q − μ)² / 2σ²) + a + b·q
fit  → μ, σ, covariance;  u_μ from the covariance
χ²_red computed against the robust per-bin noise σ (MAD of ΔI, as in attribution)
two-component variant when attribution reports overlapping candidates in the window
verdict = μ ∈ manifest window
```

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.fit_chi2_max` | 3 | reduced-χ² acceptance ceiling |
| `qa.fit_max_components` | 2 | components before declaring the window too structured to fit |

## Decision rules

### Mandatory fit guards

1. Search bounds: μ confined to the manifest window ± one window-width.
   A fit that walks out to find a better peak is answering a different
   question.
2. `χ²_red ≤ qa.fit_chi2_max` (default 3).
3. `A` above the attribution detection floor; `σ` within
   [½·bin, window width].
4. Converged (not at any bound).

### Cross-check rule (when both implementations run)

`|μ_fit − centroid| >` combined uncertainty → **soft**
`peak_position_implementations_disagree`: the peak is structured
(asymmetry, unresolved blend) — inspect the zoom plot before believing
either number. Both values go in the report regardless of which one
holds the verdict.

## When to use

Why prefer this over the centroid:

- **u_μ comes from the fit covariance** instead of a moment heuristic —
  defensible at window edges, which is exactly where verdicts get
  contested.
- **Overlapping features are separated**, not blended: each component
  gets its own μ, and the sample candidate is the one inside the window.
- **Sloped background is part of the model** rather than an edge
  interpolation.

## Trade-offs

### Why it is NOT the default

- **It asserts a functional form.** Liquid structure-factor peaks are
  asymmetric; a Gaussian μ on an asymmetric peak carries a bias the
  covariance does not report. The centroid's weaker assumptions are the
  safer default.
- **Nonlinear fits fail silently** — parameter walks, convergence at
  bounds. Every fit must pass the fit guards; a guard failure means
  fall back to the centroid implementation and record why.
- **More knobs = more verdict-laundering surface.** Model order, bounds,
  and window come from the manifest and this file's defaults — never
  from what makes the fit land.

### Failure modes / escalation

Verdict escalations identical to
[03a_peak_centroid_window.md](03a_peak_centroid_window.md)
(`main_peak_outside_window` hard, `main_peak_absent` hard,
`main_peak_borderline` soft), plus:

- **soft** `peak_fit_unreliable` — any guard failed; the verdict
  reverts to the centroid implementation, both results recorded.
- **soft** `peak_window_too_structured` — more than
  `qa.fit_max_components` needed: the window contains physics or
  parasitics the manifest doesn't describe; route the extra components
  to attribution's parasitic handling.

## Outputs

Contributes to `qa_report.json`, per curve: `peak_q`, `peak_q_unc`, `peak_sigma`,
`peak_method: "shape_fit"`, `peak_verdict`, `fit_chi2_red`, `fit_n_components`, and
`peak_q_centroid` when the cross-check ran.

## Links

Part of: [qa](../README.md). Default implementation / fallback:
[03a](03a_peak_centroid_window.md). Overlap candidates come from
[02 attribution](02_attr_feature_width.md); extra components route back to its
parasitic handling.
