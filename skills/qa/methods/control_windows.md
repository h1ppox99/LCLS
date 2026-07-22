# Control windows — signal-free q-regions as absolute per-curve witnesses

**Kind**: control check (apply when
`sample_features.<sample>.control_q_windows_A_inv` is populated;
per-curve and *absolute* — the curve-vs-curve comparison belongs to
`consistency_cross_curve.md`)

## What it checks

Per curve, per control window:

```
level  = median I in the window
slope  = robust linear fit across the window
excess = any attributed feature inside the window (by construction there should be none)
```

compared against the campaign QA history (and `qa.control_level_tolerance`).

## Rationale

Control windows are q-regions a human chose *because* sample scattering
there is flat and featureless. Whatever structure appears in them is
therefore instrument, mask, or normalization — a witness channel with
no physics in it:

- A **feature** inside a control window → parasitic scatterer or mask
  artifact; take the label from attribution rather than re-detecting.
- A **level** far off the campaign baseline → normalization or
  transmission change; check which monitor the normalization skill
  chose for this run before suspecting the sample.
- A **slope** appearing where the baseline is flat → correction-map
  error — a mis-applied solid-angle/polarization correction tilts broad
  backgrounds before it visibly moves peaks.

This check provides the absolute anchors that the peak checks
deliberately don't: peak verdicts are window-relative, so a global
intensity-scale error sails through them and is caught here.

## Parameters

| Manifest field | Default | Meaning |
|---|---|---|
| `qa.control_level_tolerance` | 0.2 | relative deviation from the campaign baseline before flagging |

If the manifest has no control windows for this sample: skip and record
(`control_windows: "skipped_not_in_manifest"`) — never invent windows
by eyeballing a flat region; that hands the witness's choice to the
thing being witnessed.

## Failure modes / escalation

- **soft** `control_window_feature` — structure where none belongs;
  attach the attribution label and q.
- **soft** `control_window_level_shift` — level outside tolerance;
  name the suspected stage (normalization choice, transmission,
  correction map), not just the ratio.
- **soft** `control_window_sloped` — significant slope in a window
  documented as flat.
- **First run of a campaign**: no baseline exists — record this run's
  values as the provisional baseline and flag `no_campaign_baseline`
  rather than fabricating a pass/fail.

## Do not

- **Do not re-choose or drop a control window because it misbehaves** —
  that discards the witness exactly when it testifies. Escalate with
  the evidence; window changes are manifest edits humans adopt.

## Contributes to `qa_report.json`

Per curve, per window: `{q_window, level, level_vs_baseline, slope, features_in_window}`.
