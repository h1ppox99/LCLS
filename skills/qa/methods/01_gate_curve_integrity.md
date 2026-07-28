---
name: xray-qa-gate-curve-integrity
description: QA integrity gate — is the curve a well-formed measurement at all? Flags malformed q-grids, coverage mismatch vs the qmap (the nm-1/A-1 units slip), starved/empty curves, and structured negative intensity. Always runs FIRST; a failing curve gets verdict invalid and no further checks.
category: qa
role: gate
gate: always — runs FIRST; a failing curve receives no further checks
status: not-wired
---

# QA · 01 — Gate: curve integrity

Is this curve a well-formed measurement at all?

## Principle

What it flags:

1. **Malformed q-grid**: non-finite or non-monotonic `q`, or bin spacing
   that jumps by more than 2× the median Δq mid-range — a concatenation
   or binning bug, not physics.
2. **Coverage mismatch**: the curve's q-range disagrees in bulk with the
   mask stage's `qmap_active_min` / `qmap_active_max`
   (`/outputs/<run_id>/mask/mask_summary.json`). A curve covering q the
   detector cannot reach is a units slip — the nm⁻¹ ↔ Å⁻¹ factor of 10
   is the classic, and it produces beautifully smooth curves whose every
   expected feature is "missing".
3. **Starved or empty curves**: NaN/empty-bin fraction above
   `qa.max_nan_bin_fraction`, or total integrated intensity
   indistinguishable from zero (integration ran on an empty selection).
4. **Negative-intensity structure**: isolated, slightly negative bins
   are normal after pedestal subtraction; *contiguous runs* of negative
   bins, or a negative median inside any feature window, mean pedestal
   or correction error upstream.

### Rationale

Every other QA method assumes the curve is a function I(q) on the
detector's reachable q-range with mostly-populated bins. Peak checks
*misdiagnose* what integrity catches: on a unit-slipped curve, the
peak-position check would report `main_peak_absent` and send a human
hunting sample problems when the bug is a factor of 10 on the q-axis.

Same philosophy as coarse selection's readout-integrity gate: these are
"the data does not exist / cannot be trusted to line up" conditions,
not quality judgments. And the same truncation rule: a curve merely
*shorter* than expected is a note; *interior* inconsistency is an
escalation.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.max_nan_bin_fraction` | 0.05 | NaN/empty fraction above which the curve is starved |
| `qa.negative_run_max_bins` | 5 | longest tolerated contiguous negative run |

Coverage tolerance: curve endpoints must lie within the qmap's active
range ± 2 bins. Requires the mask stage's summary; if it is missing,
skip the coverage item and record `coverage_check_skipped: true` — do
not reconstruct the q-range from geometry here (that re-derives the
qmap behind the mask skill's back).

## Trade-offs

Failure modes / escalation:

- **hard** `curve_q_coverage_mismatch` — bulk disagreement with the
  qmap's reachable range. Suspect units first (nm⁻¹ vs Å⁻¹, unscaled λ
  or L), then a stale qmap file from a previous run.
- **hard** `curve_integrity_failed` — malformed grid, empty curve, or
  structured negative intensity.
- **soft** `curve_statistics_starved` — high NaN fraction concentrated
  at extreme q is expected (sparse rings; the mask refinements already
  exclude rings with < ~50 pixels). NaN gaps in the *interior* are not —
  report which q-ranges.

## Outputs

Contributes to `qa_report.json`, per curve: `integrity_verdict`, `nan_bin_fraction`,
`q_range` vs `qmap_active_range`, `negative_run_max`, plus the reason
string for any `invalid` verdict.

## Links

Part of: [qa](../README.md). Runs before [02 attribution](02_attr_feature_width.md).
Consumes the mask stage's `mask_summary.json`; a glitch density > ~5 % of bins in
[04b](04b_quality_pointwise_glitch.md) reroutes to this gate's `curve_integrity_failed`.
