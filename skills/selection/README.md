---
name: xray-selection
description: Selection skill category — deciding which shots/events enter the sum. Two AGENT-DECIDED shot-quality cuts on the ipm2 flux monitor (low-flux exclusion 01a, bright-tail exclusion 01b, split on purpose because they have different physics), plus an appended per-shot Poisson-conformity screen (02) that drops shots for statistical nonconformity, never for brightness. Selection acts on whole shots - never on pixels (masking), never rescaling (normalization).
category: selection
role: category-index
status: wired
---

# Selection (category index)

**Goal.** Decide which shots enter the accumulation. Selection removes *whole
shots/events*; it never removes pixels (that is [masking](../masking/README.md))
and never rescales intensities (that is [normalization](../normalization/README.md)).

## Methods

One file each, wired into this pipeline:

| File | Method | Decision character |
|---|---|---|
| [01a_low_ipm_exclusion.md](01a_low_ipm_exclusion.md) | Low-ipm2 (near-zero-flux) shot exclusion | **agent decides** run/skip + threshold — mandatory before any per-shot division |
| [01b_high_ipm_exclusion.md](01b_high_ipm_exclusion.md) | High-ipm2 (bright-tail) shot exclusion | **agent decides** run/skip + percentile — conditional/protective, skipping is legitimate |
| [02_shot_poisson_conformity.md](02_shot_poisson_conformity.md) | Per-shot Poisson-conformity screen (appended; needs frames) | **agent decides** run/skip + fence + per-class routing — drops for nonconformity, never for brightness |

The two ipm cuts read the same evidence (shot_table `ipm2` + EVR flags) but are
**separate decisions with different physics** — noise amplification on the low
side, saturation/nonlinearity on the high side. Do not couple them. Method 02
is an **append** on the survivors (01a → 01b → 02): it needs per-shot frames,
tests each shot against the conditional Poisson null of
[qa/08](../qa/methods/08_pixel_photon_statistics.md) transposed to the shot
axis, and adds specificity the monitor-only thresholds cannot — on Run0475 it
flags 7 mid-range pointing-excursion shots p99 never looks at while clearing
all 28 above-p99 bright shots. The always-on baseline (`require_xray_on`: drop
EVR-dropped shots) is part of the `decisions.json` selection block, not a
separate method file.

The parent X-ray skill set carries further selection methods not wired into
this pipeline (event temporal alignment, good-pixel selection, q-range
restriction, correlation gate); step 0's deterministic global time-sort already
covers event ordering here.

## Golden rules

1. **A mask is not a selection** — never fold flux dropouts into the pixel mask
   or vice-versa.
2. **Selection precedes normalization**: [normalization/01](../normalization/01_per_shot_flux_ipm2.md)
   must never see excluded shots (near-zero monitor readings are noise, often
   negative — a divide guard does not catch −0.0002).
3. **Report what each cut removed** — counts and percentages in the decision log;
   silent truncation reads as "kept everything" when it did not.
