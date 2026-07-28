---
name: xray-qa-consistency-cross-curve
description: QA consistency check — do this run's curves tell one story? Pairwise level ratios in control regions, main-peak centroid spread in joint-uncertainty units, and quality-rank outliers, on a common q-grid. The last gate against population-dependent artifacts that are invisible per-curve but fatal in difference signals. Runs LAST, when the run produced 2+ curves.
category: qa
role: consistency
gate: apply when the run produced ≥ 2 curves; runs LAST — consumes the other checks' per-curve results
status: not-wired
---

# QA · 07 — Consistency: cross-curve

Do this run's curves tell one story? Runs LAST — it consumes the other checks'
per-curve results.

## Principle

What it checks:

All pairs of curves, on a common q-grid (interpolate the coarser grid
to the finer; compare only over the shared q-range; exclude bins either
curve has as NaN):

```
level ratio     — in control windows if defined, else full range minus
                  all attributed-feature supports
centroid spread — main-peak positions across curves, in joint-uncertainty units
quality spread  — one curve much rougher than its siblings (smoothness ranks)
```

### Rationale

Condition-split curves (e.g. pump-on vs pump-off populations) are where
the science lives, and difference signals are small — so
*population-dependent* artifacts are the most dangerous kind: invisible
in any single curve, fatal in the difference.

- **Level ratio ≠ 1** beyond tolerance → per-condition normalization
  error. A 1 % level mismatch swamps a sub-percent difference signal;
  this is the failure mode the whole normalization stage exists to
  prevent, checked at the last gate before a human reads the curves.
- **Centroid spread** beyond joint uncertainty → real condition physics
  OR condition-correlated systematics. QA cannot distinguish the two —
  flagging it *before* someone reads the difference curve as physics is
  the point. State both interpretations in the report.
- **One rough sibling** → that population is starved or contaminated;
  attach its kept-shot count from the selection summary before anyone
  blames the detector.

Division of labor: [05_control_windows.md](05_control_windows.md) judges each curve
*absolutely* against the manifest and campaign baseline; this check judges curves
*against each other* and needs no manifest windows to run.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.cross_curve_level_tolerance` | 0.01 | pairwise level-ratio deviation from 1 in control regions |
| `qa.cross_curve_centroid_max_sigma` | 3 | centroid-spread fence, joint-σ units |

## Trade-offs

Failure modes / escalation:

- **hard** `cross_curve_level_mismatch` — beyond tolerance in control
  regions. Hard because any difference curve built on it is
  quantitatively wrong, and QA is the last gate before publication of
  the run's results.
- **soft** `cross_curve_centroid_spread` — with both interpretations
  (physics vs systematics) stated explicitly.
- **soft** `cross_curve_quality_outlier` — with the suspect
  population's selection statistics attached.
- **Interpolation trap**: curves binned differently by design can fake
  a level mismatch through interpolation error near sharp features —
  exclude attributed-feature supports from the level comparison (they
  are excluded above for exactly this reason) and record both grids.

## Do not

- **Do not average disagreeing curves to "stabilize" the result.** A
  cross-curve disagreement is information about the run; burying it in
  a mean launders the artifact into the science.

## Outputs

Contributes to `qa_report.json`, run-level: `cross_curve` — pairwise
`{curves, level_ratio, centroid_delta_sigma, verdict}`, plus
`quality_outliers`.

## Links

Part of: [qa](../README.md). Absolute counterpart:
[05 control windows](05_control_windows.md). Feature supports excluded from level
comparison come from [02 attribution](02_attr_feature_width.md); quality ranks from
[04a](04a_quality_smoothness_curvature.md).
