---
name: xray-verification
description: Verification skill category — quantitative acceptance criteria for the pipeline endpoint (masked, normalized, azimuthally-averaged I(q)). Each criterion is its own md file with a machine-readable threshold block; the VERIFIER agent reads every criterion, judges the measured metrics against it, and on failure writes actionable feedback targeted at the responsible upstream phase (reduction or mask).
category: verification
role: category-index
status: wired
---

# Verification (category index)

**Goal.** Decide whether `outputs/<run>/masked_sum.npy` is scientifically usable, using the
deterministic metrics computed by `pipeline/step4_iq.py` (`iq_metrics.json` + `iq.png`).
The verifier never recomputes physics by hand and never edits upstream files — it *judges*
and it *writes feedback*.

## Methods

| File | Criterion | Catches |
|---|---|---|
| [01_iq_quality.md](01_iq_quality.md) | I(q) ring presence, contrast, background sanity, coverage | wrong selection/normalization (washed-out rings), missing mask layers (negative dips, bumps, noise), over-masking |

More criteria will be added as separate files (peak-width / resolution, azimuthal
uniformity, cross-run consistency, ...). One criterion = one file, same as the
masking/selection/normalization skill sets. (`iq_metrics.json` already carries an
informational `azimuthal_uniformity` scan of the endpoint — no criterion reads it yet.)

## Contract

The verifier writes `outputs/<run>/verify_report.json`:

```json
{
  "iteration": 1,
  "overall": "pass | fail",
  "criteria": [
    {"id": "iq_rings", "pass": true,
     "measured": {"...": "numbers actually observed"},
     "threshold": {"...": "numbers required"},
     "note": "one-line judgement"}
  ],
  "feedback": {
    "target_phase": "reduction | mask",
    "message": "concrete, actionable instruction with numbers"
  }
}
```

`feedback` is `null` when `overall` is `pass`. When failing, pick the **most likely
responsible phase**: washed-out or missing rings and wrong background level usually
trace to selection/normalization (→ `reduction`); negative dips, localized bumps,
excess background noise, or coverage loss usually trace to masking (→ `mask`).
The feedback message is injected verbatim into that phase's prompt on the next
iteration — write it for the agent that will have to act on it.

## Golden rules

1. **Judge against numbers, not vibes.** Every pass/fail cites measured vs threshold.
2. **One failure is enough to fail the run**, but report every criterion's result.
3. **Feedback must be actionable**: name the metric, the measured value, the threshold,
   and the concrete change to try (e.g. "beamstop band rows 970-1006 appears unmasked:
   neg_bin_fraction 0.31 > 0.02 — add the horizontal shadow band layer").
4. **Do not weaken thresholds to pass a run.** Thresholds change only by editing the
   criterion file itself (a human-reviewed act), never inside a verifier run.
