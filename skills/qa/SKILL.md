---
name: qa
description: Judge whether 1D radial averages from an LCLS/XPP run are physically reasonable. Reads expected peak windows, known instrument backgrounds, and calibration tolerances from the manifest at /data/manifest.yaml. Checks are organized as an integrity gate, feature attribution, and per-evidence check families — variants documented in methods/*.md with selection rules below. Use after azimuthal integration has produced 1D curves.
---

# Quality assurance

## When to use

After integration has produced one or more `.npz` curves in
`/outputs/<run_id>/curves/`. QA validates them physically and writes a
structured report.

**QA judges; it never fixes.** No upstream output is modified by this
skill — a failing check routes back as an escalation to the stage that
owns the cause, or to a human.

## Source of truth

All campaign-specific values come from the manifest at **`/data/manifest.yaml`**
(see `common` SKILL for usage). Don't hardcode peak windows. Read them.

Fields this skill reads:

- `sample_features.<sample>.main_peak_q_window_A_inv` — expected peak window
  for the sample being processed. The orchestrator passes the sample name.
- `sample_features.<sample>.control_q_windows_A_inv` — control regions for
  background checks.
- `instrument_backgrounds` — broad non-Bragg features (e.g., Kapton window)
  that QA should label and not flag as failures.
- `calibration_tolerance.lab6_max_delta_q_pct`,
  `calibration_tolerance.per_run_drift_rings_to_check`,
  `calibration_tolerance.per_run_drift_check_required` — drift check.
- `qa.*` — check configuration (see proposed block below). If absent,
  use the defaults stated in each method file and record
  `manifest_block_missing: true`.

Proposed manifest block (add to `manifest_template.yaml` when adopted):

```yaml
qa:
  # Peak-position implementation: "centroid_window" (default) or "shape_fit".
  # Exactly one produces the verdict; the other may run as a recorded cross-check.
  peak_method: centroid_window
  # Curve-quality lenses to run; global + localized compose.
  quality_checks: [smoothness_curvature, pointwise_glitch]
  # Feature-attribution width boundaries (Å⁻¹): sharp ≤ bragg, diffuse ≥ diffuse,
  # in between = ambiguous.
  bragg_sigma_max_A_inv: 0.02
  diffuse_sigma_min_A_inv: 0.04
  feature_min_prominence_sigma: 5
  # Integrity limits.
  max_nan_bin_fraction: 0.05
  negative_run_max_bins: 5
  # Glitch fences.
  glitch_window_bins: 11
  glitch_k: 6
  # Control-window and cross-curve tolerances.
  control_level_tolerance: 0.2
  cross_curve_level_tolerance: 0.01
  cross_curve_centroid_max_sigma: 3
```

## Available methods

Method files live in `methods/`. Each contributes named fields to
`qa_report.json`; the report is the union of what ran.

| File | Role | What it checks |
|---|---|---|
| `methods/gate_curve_integrity.md` | gate (always, FIRST) | well-formed q-grid, coverage vs qmap, starved bins, negative structure |
| `methods/attr_feature_width.md` | attribution (always, SECOND) | labels every feature: sample / calibration / instrument background / parasitic — by width |
| `methods/peak_centroid_window.md` | peak-position (default impl.) | sub-bin centroid of the main peak vs the manifest window |
| `methods/peak_shape_fit.md` | peak-position (alternative impl.) | model fit — for overlapping features and borderline verdicts |
| `methods/quality_smoothness_curvature.md` | curve-quality · global lens | whole-curve roughness (second-derivative L2, the legacy metric) |
| `methods/quality_pointwise_glitch.md` | curve-quality · localized lens | single-bin spikes/notches/steps, attributed to upstream causes |
| `methods/control_windows.md` | control (gated on manifest windows) | signal-free regions as absolute normalization/mask witnesses |
| `methods/calib_lab6_drift.md` | calibration (gated on run type) | per-ring Δq/q vs LaB6 prediction; residual-pattern diagnosis |
| `methods/consistency_cross_curve.md` | consistency (gated on ≥ 2 curves, LAST) | curves agree with each other — levels, centroids, quality |

Selection rules:

1. **The gate and attribution always run, in that order.** Attribution's
   labels are inputs to everything downstream — the glitch lens without
   them reports Bragg physics as artifacts; the peak check without them
   can lock onto a parasitic.
2. **Peak-position implementations are alternatives — exactly one
   produces the verdict** (`qa.peak_method`). Default `centroid_window`;
   `shape_fit` per its "when to prefer" section. The other may run as a
   cross-check, and disagreement between them is itself reportable
   (`peak_position_implementations_disagree`). Choose the implementation
   BEFORE seeing either verdict — switching after the fact is verdict
   laundering.
3. **Curve-quality lenses compose.** Global (smoothness) and localized
   (glitch) are two lenses on one family — run what
   `qa.quality_checks` lists; they share no threshold.
4. **Gated checks skip loudly, never silently.** Control windows
   (manifest has them), drift check (required + rings visible),
   cross-curve (≥ 2 curves) — a skipped check is recorded with its
   reason; the report skill compares checks-run across runs.
5. **One estimator per run** for all sub-bin peak positions (verdict
   peak AND LaB6 rings) — mixing estimators mixes their systematics
   into the drift residuals.

## Procedure

1. Load each curve; run `gate_curve_integrity` — invalid curves are
   reported and excluded from everything below.
2. Run `attr_feature_width` on each surviving curve.
3. Per curve: the configured peak-position check, the configured
   quality lenses, `control_windows` if manifest-gated in,
   `calib_lab6_drift` if required and rings are visible.
4. Run `consistency_cross_curve` across surviving curves.
5. Assemble `qa_report.json` / `qa_report.md`; zoom plots for every
   borderline or escalated verdict.

## Do not

- **Don't tune geometry to make a peak land in window.** That's the
  load-bearing mistake the manifest exists to prevent. If the peak is
  outside the manifest's window with the manifest's geometry, that IS
  the result — escalate (geometry investigation, or a manifest window
  update backed by a clean calibration run).
- **Don't launder verdicts** — no re-running with a different peak
  implementation, wider windows, or coarser binning until a check
  passes. Method choices are recorded with reasons before verdicts are
  computed.
- **Don't gate calibration-run curves on smoothness** — sharp physics
  scores rough by construction; compare like with like.
- **Don't modify anything upstream** — masks, selections, curves, and
  the manifest are read-only to this skill.

## Outputs

Write to `/outputs/<run_id>/qa/`:
- `qa_report.md` — human-readable per-curve verdict table + escalations.
- `qa_report.json` — structured: `{pass_n, total, rows: [...], checks_run,
  checks_skipped, escalations}`. Row fields are documented per method
  file under "Contributes to qa_report.json".
- Per-curve zoom plots for borderline or escalated verdicts.
