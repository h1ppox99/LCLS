---
name: xray-selection-high-ipm-exclusion
description: High-flux (large ipm2) shot exclusion — an AGENT-DECIDED step. The agent checks for detector/monitor nonlinearity at the bright end (ratio droop, gain-mode switching, saturation) and decides whether to cut the bright tail and at what percentile. On Run0475 the evidence for harm is weak (linear to p99, no signal-driven gain switching), so this cut is protective / baseline-compatibility rather than evidence-driven — skipping is a legitimate outcome.
category: selection
role: agent-decided shot cut (whole shots) — bright-tail side
gate: conditional/protective — evidence of nonlinearity, or baseline compatibility; skipping is legitimate
status: wired
---

# Selection · 01b — High-ipm2 exclusion (agent decides)

**Split from** the old `01_shot_flux_quality.md`: the bright-side cut has *different physics*
(saturation/nonlinearity) than the low-side cut (noise amplification) and deserves its own
decision.

## The decision the agent owns

> **Cut the bright tail or not? And if yes, at which percentile / absolute level?**
>
> Unlike the low cut, the case for this one is *conditional*: it protects against
> nonlinearity that may or may not be present. The agent must look for actual evidence of
> harm before spending statistics — the brightest shots carry the most photons.

## Principle

Very bright shots can exceed the linear range of (a) the detector front-end — pixels start
switching gain mode (G0→G1) mid-signal, (b) the intensity monitor itself, or (c) downstream
assumptions (e.g. one-photon counting). If any of that happens, the brightest shots distort
the flux-referenced result and should be dropped (or handled specially). If none of it
happens, cutting them only throws away the highest-SNR frames.

## Decision rules

### Evidence to gather before deciding

1. **Ratio flatness at the top**: bin `det_total / ipm2` by ipm2; a droop in the top bins
   = detector or monitor rolling over. (Run0475: flat at ~22×10⁻⁶ from ipm2 ≈ 3700 all the
   way to the p99 cut — **no droop**.)
2. **Gain-mode switching on bright shots**: fraction of pixels with gain bits ≠ G0 among
   the brightest shots. (Run0475: switching pixels are a fixed set of 566 stuck/bad pixels;
   even 12 597-photon shots stay in G0 — LaB6 signal is ~10⁻² photons/pixel, orders below
   the ~34-photon G0 headroom.)
3. **Monitor spec/saturation**: does ipm2 clip or curve at the observed maxima?
4. **Baseline compatibility**: the human baseline (Taekeun) rejected the top 1 % (`p99`)
   before its 100-shot calibration sum. Reproducing that baseline requires the same cut.

### Run / skip / cut placement

**Decide RUN when any of these hold**
- The det/ipm2 ratio droops in the top percentiles (nonlinearity observed);
- signal-driven gain switching appears on bright shots (not just stuck pixels);
- the monitor saturates below the brightest observed values;
- the task is to **reproduce the human baseline** → use its exact convention (p99).

**Decide SKIP when all of these hold**
- Linearity holds to the top (Run0475: it does);
- no signal-driven gain switching (Run0475: none);
- statistics are precious and the product benefits from the brightest frames.

**Cut placement (if RUN)**
- Percentile form (`p99`, `p99.5`) adapts to run brightness; absolute form is better when
  the physical saturation level is known. Record which convention and why.

### Operator hint — CURRENT ROUND (overrides the run/skip choice)

> For this round, **RUN the high cut at percentile 99** (drop shots above the
> 99th percentile of ipm2 among x-ray-on shots), regardless of the linearity
> evidence — this matches the human baseline convention. Still record the
> linearity numbers you measured alongside the decision.

## Evidence (Run0475)

Worked example — the verdict: evidence-driven answer: **SKIP is defensible** (32 shots
above p99 are linear and clean); baseline-compatibility answer: **RUN at p99** to match
Taekeun's sum. The agent must pick based on the goal of the current task and *say so* in
the decision log — this is exactly the kind of judgment this skill set exists to make
explicit.

## Outputs

Log: run/skip, convention (percentile/absolute), shots rejected, and rationale, e.g.
"reproducing baseline → p99 (32 shots dropped)" or "linearity verified to max → no cut".
Executed via `decisions.json → selection`.

## Links

Part of: [selection](README.md). Companion: the low-side cut is a separate decision —
[01a_low_ipm_exclusion.md](01a_low_ipm_exclusion.md).
