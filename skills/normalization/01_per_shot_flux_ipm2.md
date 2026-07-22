---
name: xray-normalization-per-shot-flux
description: Per-shot incident-flux (ipm2) normalization — an AGENT-DECIDED step. The agent runs a cheap bin-consistency test (chronological bins, CV with vs without normalization) and decides whether to normalize, with which monitor, and in which form (per-shot division vs ratio-of-sums). Evidence on Run0475 - bin-to-bin CV 6.03%→3.55%; decision was NORMALIZE. Mandatory prerequisite - low-ipm exclusion. Includes the harmonic-vs-arithmetic-mean reference subtlety.
---

# Normalization · Method 1 — Per-shot incident-flux (ipm2) normalization (agent decides)

**Acts on:** shot ↔ shot. **Part of:** [normalization](README.md). **The load-bearing one.**

## The decision the agent owns

> **Normalize or not? With which monitor? In which form?**
>
> Do not apply this step by reflex, and do not skip it by reflex. The agent runs the
> evidence test below (seconds, smalldata-only), decides, and records the decision.
> On Run0475 the evidence said **normalize** — but the point of this file is the test,
> not the answer.

## Principle

The FEL's per-shot incident flux `I0` fluctuates ~100 % shot-to-shot and drifts over
minutes. The physical observable is scattering *per incident photon*, so each shot is
divided by its own flux before combining:

```
n_i  = ipm2c_i / ⟨ipm2c⟩_kept        # offset-corrected, dimensionless, mean 1
S(q) = Σ_i f_i(q) / n_i              # per-shot division   (needs linear-plateau shots only)
S(q) = Σ_i f_i(q) / Σ_i n_i          # ratio-of-sums form  (robust to monitor noise)
```

`ipm2c = ipm2 − offset`, offset = median ipm2 over x-ray-off shots (Run0475: +4.5).

## The evidence test (run this before deciding)

1. Keep only shots passing [01a low-ipm exclusion](../selection/01a_low_ipm_exclusion.md)
   at the linear-plateau threshold.
2. Split kept shots into ~8 **chronological** bins (mimics how delay scans sample time).
3. Compute each bin's mean I(q); compare bin-to-bin CV of total intensity, with vs
   without dividing each shot by `n_i`.
4. Sanity anchors: correlation of per-shot detector total with the monitor; ratio
   flatness across the intensity range.

### Run0475 results (worked example)

| Test | unnormalized | normalized by ipm2 |
|---|---|---|
| chronological bins (the realistic case) | CV 6.03 % | **3.55 %** |
| intensity-sorted bins (pure-flux control) | 61.5 % | 2.15 % |
| random bins (drift hidden — underestimates value) | 2.99 % | 2.42 % |
| per-shot CV, brighter half | 63 % | 45 % (identical to sample_diode's 45 %) |

det~ipm2 r = 0.841 ≈ det~sample_diode r = 0.840; Poisson floor for these bins ≈ 0.09 %,
so the removed spread was systematic drift, not counting noise.
**Decision on Run0475: NORMALIZE** (flux-driven half of the drift removed; residual
~3.5 % is pointing/spectral drift that no monitor division can remove — that residual is
what pump-on/off referencing is for).

## Decision rules

**Decide NORMALIZE when any of these hold**
- The bin test shows a material CV drop (it did: 6.0 → 3.5 %);
- subsets of shots will be compared (delay bins, on/off groups, run-to-run) —
  for pump–probe this is **mandatory**, difference signals are ≪ the 6 % drift;
- monitor correlates with the detector total (r ≳ 0.8) over the kept shots.

**Decide SKIP when all of these hold**
- The only product is a single static sum used for geometry/masking (normalization
  leaves the shape unchanged — per-pixel corr 0.985 on Run0475);
- or the monitor is broken/uncorrelated on this run (dividing by a bad monitor *injects*
  noise — this is why the test comes first, and why skipping must be possible).

**Form and monitor choice (if normalizing)**
- Shots restricted to the monitor's linear plateau → per-shot division is safe;
  otherwise prefer ratio-of-sums.
- ipm2 (upstream) vs sample_diode: statistically equivalent here (45 % vs 45 %), but
  ipm2 is upstream of the sample — safe for pump–probe, whereas a post-sample diode can
  divide out the pump-induced signal itself. Default: ipm2. Alternative reference
  parameters (upstream GMD/BMMON, in-hutch diode boxes, self-normalization) each have
  their own method file — see the [monitor menu](README.md) comparison table.

## Hard prerequisite

[Low-ipm exclusion](../selection/01a_low_ipm_exclusion.md) **must run first**: near-zero
shots have monitor readings that are pure noise (often negative — 321/394 low shots on
Run0475 had sample_diode ≤ 0); dividing by them destroys bins. A guard against NaN/0 is
not sufficient — it does not catch −0.0002.

## Reference-convention subtlety (important)

`Σwf/Σw` with `w = ⟨ipm2⟩/ipm2` is algebraically referenced to the **harmonic-mean** flux;
a plain mean references the **arithmetic mean**. On Run0475 this is a constant ×0.63
rescale (HM/AM = 5213/8264), *not* a physical difference — multiply by AM/HM to compare.

## Record the decision

Log: normalize yes/no, monitor, form, offset used, kept-shot definition, the bin-test
numbers that justified it, and the residual CV (hand it to the QA/differencing stage).

## Repo

`weighted_sum_v2.py`, `Normalization_method.md`, `Normalization_方法与对比.md`,
`run475_weighted_sum.npy`, `run475_pershot_weights.npy`.
