---
name: xray-selection-low-ipm-exclusion
description: Low-flux (small ipm2) shot exclusion — an AGENT-DECIDED step. The agent examines the ipm2 distribution and downstream plan, then decides whether to exclude near-zero-flux shots and where to place the threshold. Mandatory before any per-shot flux normalization (division by a near-zero, noise-dominated monitor value explodes); optional but recommended for plain sums (pure SNR gain). Grounded in Run0475 evidence.
---

# Selection · Method 1a — Low-ipm2 exclusion (agent decides)

**Level:** whole shots. **Part of:** [selection](README.md).
**Split from** the old `01_shot_flux_quality.md`: low-side and high-side cuts are separate
decisions with different physics — do not couple them.

## The decision the agent owns

> **Run this exclusion or not? And if yes, where does the threshold go?**
>
> This is not a fixed pipeline stage. The agent gathers the evidence below, decides, and
> records the decision with its rationale. Skipping is a valid outcome — but must be an
> *explicit, recorded* choice, never a default.

## Principle

Shots with near-zero incident flux carry detector readout noise but almost no photons.
Adding them to a sum adds variance without signal; dividing by their monitor value
(normalization) amplifies noise unboundedly, because at zero beam the monitor reading is
its own electronic noise — small, often **negative**.

## Evidence to gather before deciding (all cheap, smalldata-only)

1. **ipm2 histogram** — look for a near-zero pile separated by a gap from the main body.
2. **`lightStatus/xray`** — dropped (x-ray off) shots always go; the question is the
   remaining "nominally on but near-zero" shots.
3. **Monitor zero offset** — median ipm2 over x-ray-off shots (Run0475: +4.5, i.e. 0.12 %
   of the median; subtract before any thresholding or ratio).
4. **Photon-yield spot check** — photon pixels (7–12 keV) per shot for a few low vs
   normal shots.
5. **Downstream plan** — will any step divide by a per-shot monitor (normalization,
   weighting)? If yes, this exclusion stops being optional.

## Evidence from Run0475 (why the default is RUN)

394 shots (12.3 % of the run) had xray on but ipm2 < 200:

| Metric | low group (394) | normal group (394 sampled) |
|---|---|---|
| photon pixels / shot (median) | **41** (mostly fake — fixed hot-ish pixels) | **6448** |
| total photon yield | 0.73 % of the normal group | — |
| sample_diode | median **−0.0002**; ≤ 0 in **321/394** | +0.0019 |
| group mean I(q) | no rings; systematic negative high-q tail (pedestal drift) | LaB6 rings |

Removing them: signal −0.3 %, noise variance −12.3 % → **~6 % SNR gain for free**, and it
removes a *systematic* (q-dependent) contamination, not just white noise.

## Decision rules

**Decide RUN when any of these hold**
- The histogram shows a near-zero pile (it does on Run0475);
- any downstream step divides by ipm2 or sample_diode → **mandatory** (321/394 low shots
  have diode ≤ 0; one such shot can destroy a whole bin);
- per-shot ratios/weights will be formed → threshold should move up to the start of the
  monitor's **linear plateau** (Run0475: det/ipm2 ratio is flat only above ipm2 ≈ 3000;
  below, the ratio is biased +50 %).

**Decide SKIP when all of these hold**
- Distribution is unimodal with no near-zero pile, AND
- the only product is an unweighted diagnostic sum (shape barely affected), AND
- statistics are so scarce that a 12 % frame loss matters more than a 6 % SNR gain.

**Threshold placement (if RUN)**
- Plain sums: anywhere in the gap between pile and body (Run0475: 200 — insensitive).
- Before per-shot division: start of the linear plateau (Run0475: ≈ 3000 ≈ 0.8×median).
- Always threshold on offset-corrected ipm2.

## Operator hint — CURRENT ROUND (overrides threshold placement)

> For this round, exclude **only the near-zero pile: threshold ipm2c < 200**
> (the gap between pile and body). Do **not** raise the threshold to the linear
> plateau — keep the 200–3000 mid-band shots in the dataset. If you also decide
> to normalize, reconcile the form yourself: per-shot division outside the
> plateau violates the normalization skill's linearity rule, so consider
> `ratio_of_sums` (or justify why not). Record that reasoning.

## Record the decision

Log: run/skip, threshold, shots kept/rejected (counts and %), and the one-line rationale
(e.g. "near-zero pile of 418 shots at <200; normalization planned → excluded"). Silent
truncation is the enemy of reproducibility.

## Feeds

[normalization § per-shot-flux](../normalization/01_per_shot_flux_ipm2.md) (which must not
see excluded shots) · the correlation gate (parent X-ray skill set, not wired here).

## Companion

High-side cut is a separate decision: [01b_high_ipm_exclusion.md](01b_high_ipm_exclusion.md).
