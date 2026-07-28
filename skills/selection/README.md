---
name: xray-selection
description: Selection skill category — deciding which shots/events enter the sum. Wired into XTC_Agent - two AGENT-DECIDED shot-quality cuts on the ipm2 flux monitor (low-flux exclusion 01a, bright-tail exclusion 01b), split on purpose because they have different physics. Selection acts on whole shots - never on pixels (masking), never rescaling (normalization).
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

The two cuts read the same evidence (shot_table `ipm2` + EVR flags) but are
**separate decisions with different physics** — noise amplification on the low
side, saturation/nonlinearity on the high side. Do not couple them. The
always-on baseline (`require_xray_on`: drop EVR-dropped shots) is part of the
`decisions.json` selection block, not a separate method file.

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
