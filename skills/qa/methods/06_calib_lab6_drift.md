---
name: xray-qa-calib-lab6-drift
description: QA calibration check — per-run geometry witness comparing sub-bin LaB6 ring positions against lattice predictions (|dq/q| vs manifest tolerance); the residual PATTERN across rings localizes the error (L/lambda, tilt, beam center, misidentified ring). The hard-escalation authority on q-scale. Never tune geometry to make rings land.
category: qa
role: calibration
gate: apply when calibration_tolerance.per_run_drift_check_required is true AND at least one LaB6 reflection is visible
status: not-wired
---

# QA · 06 — Calibration: LaB6 drift

Per-run geometry witness against predicted ring positions — the hard-escalation
authority on q-scale.

## Principle

What it checks:

For each ring in `calibration_tolerance.per_run_drift_rings_to_check`:

```
q_pred = 2π · √(h² + k² + l²) / 4.15695 Å        (LaB6, Pm-3m)
q_obs  = sub-bin position by the SAME estimator the peak-position
         family used this run (centroid or fit — one estimator per run,
         or the residuals mix two systematics)
per ring:  |Δq / q_pred| · 100 ≤ calibration_tolerance.lab6_max_delta_q_pct
```

Rings must first be identified as such by attribution (sharp + at a
LaB6 prediction) — a parasitic spot near a predicted q must not silently
stand in for the ring.

### Three call sites, one contract

This is the same check as mask SKILL step 5 and the pyFAI bridge's
validation gate — same tolerance, same prediction, same sub-bin
refinement. QA's instance is the *last word* because it sees the final
curves: if the mask-stage check passed and QA's fails, whatever sits
between them (correction map, integration) moved the rings. Record
both results side by side — the *location* of the disagreement is the
diagnosis.

## Parameters

None of its own — `calibration_tolerance.lab6_max_delta_q_pct` and
`per_run_drift_rings_to_check` come from the manifest; the lattice
constant is physics, not configuration.

## Decision rules

### The residual PATTERN is the diagnosis

Report per-ring residuals, then read their structure — it localizes the
error for the human:

| Pattern across rings | Points at |
|---|---|
| common multiplicative offset (Δq/q same sign and size) | distance L or wavelength λ wrong |
| offset growing/shrinking systematically with q | geometry model (tilt, small-angle breakdown) |
| residuals centered but rings broadened vs campaign norm | beam-center error — 1D averaging smears a decentered ring symmetrically without moving its centroid much |
| single ring off, others clean | misidentified ring or overlapping parasitic — go back to attribution before believing it |

## Trade-offs

Failure modes / escalation:

- **hard** `lab6_ring_delta_q_exceeds_tolerance` — the established
  name; attach per-ring residuals and the pattern reading.
- **soft** `lab6_rings_not_visible` — check required but no reflection
  found. Legitimate on sample runs without residual LaB6 scatter;
  record the skip, never fake a pass.
- **soft** `drift_trend_systematic` — all residuals same sign but
  within tolerance: an early warning for the run table before it
  becomes a hard failure next run.

## Do not

- The load-bearing rule, verbatim from every layer of this pipeline:
  **do not tune L, beam center, or λ to make rings land.** Drift
  detected honestly is operational information; drift absorbed silently
  corrupts every curve downstream of the geometry.

## Outputs

Contributes to `qa_report.json`, per curve: `lab6_ring_residuals` —
`{hkl, q_pred, q_obs, delta_q_pct}` per ring, `drift_pattern`, `drift_verdict`.

## Links

Part of: [qa](../README.md). Rings are identified by
[02 attribution](02_attr_feature_width.md); sub-bin positions use the run's
peak-position estimator ([03a](03a_peak_centroid_window.md) /
[03b](03b_peak_shape_fit.md)) — one estimator per run.
