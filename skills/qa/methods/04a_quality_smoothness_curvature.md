---
name: xray-qa-quality-smoothness-curvature
description: QA curve-quality check, global lens — whole-curve roughness as the second-derivative L2 norm over intensity range (the legacy qa_check.py metric, kept unchanged for campaign comparability). Relative metric - rank within run and against campaign history; never gate calibration-run curves on it. Composes with the localized glitch lens 04b.
category: qa
role: curve-quality (global lens)
gate: run what qa.quality_checks lists; composes with 04b — same family, orthogonal lens
status: not-wired
---

# QA · 04a — Curve quality: smoothness

Global second-derivative L2 norm. Composes with
[04b_quality_pointwise_glitch.md](04b_quality_pointwise_glitch.md) — same family,
orthogonal lens.

## Principle

What it measures:

```
smoothness = ‖Δ²I‖₂ / (max I − min I)        (NaN bins excluded)
```

Lower = smoother. This is the existing `qa_check.py` metric kept
unchanged, so QA history stays comparable across the campaign.

### What it catches

Whole-curve roughening — the signature of causes that touch every bin:

- residual unmasked hot pixels sprinkled across many rings (each pokes
  its own q-bin);
- binning finer than the statistics support;
- a noisy normalization channel amplifying shot-to-shot variance into
  every bin.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.smoothness_max` | unset | optional hard ceiling once a campaign baseline exists |

## Decision rules

Interpretation rules:

- **Relative metric.** Compare across a run's curves and against the
  campaign's QA history; an absolute threshold from nowhere is noise.
  The manifest may pin `qa.smoothness_max` once a campaign baseline
  exists; until then, record and rank, and set
  `manifest_block_missing: true`.
- **Run-type awareness.** Sharp physics IS curvature: LaB6 calibration
  curves score high by construction. Never gate a calibration-run curve
  on smoothness; compare like with like (calibration vs calibration,
  sample vs sample).

## Trade-offs

### Known weakness (why the glitch lens exists)

A global L2 dilutes a single-bin spike over ~10³ bins — a curve can
pass smoothness while carrying one glaring artifact. Localized defects
are [04b_quality_pointwise_glitch.md](04b_quality_pointwise_glitch.md)'s job.
Treat the two as one family with two lenses, not as alternatives: this lens ranks
*overall* statistical health; that one finds and *attributes* individual defects.

### Failure modes / escalation

- **soft** `smoothness_out_of_band` — worst-in-run by a wide margin, or
  above the pinned ceiling. Smoothness is a *symptom* metric: the
  escalation should point at the suspected cause (mask refinement
  summary, normalization choice, binning), not just report the number.
- Never hard on its own — the causal stage's checks own the hard
  verdicts.
- **Range-normalization trap**: a curve dominated by one huge peak has
  a large `max − min`, deflating the score of a rough baseline. When
  ranking, also record the score computed with the main-peak window
  excluded (`smoothness_off_peak`).

## Outputs

Contributes to `qa_report.json`, per curve: `smoothness`, `smoothness_off_peak`,
`smoothness_rank_in_run`.

## Links

Part of: [qa](../README.md). Localized sibling lens:
[04b](04b_quality_pointwise_glitch.md). Legacy metric source: `scripts/qa_check.py`.
Main-peak window for `smoothness_off_peak` comes from
[02 attribution](02_attr_feature_width.md).
