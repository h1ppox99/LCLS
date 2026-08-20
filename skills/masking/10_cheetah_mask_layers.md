---
name: xray-masking-cheetah-mask-layers
description: Literature method (Barty et al. 2014, J. Appl. Cryst. 47, 1118 — Cheetah) — the production XFEL practice of SEPARATE, PURPOSE-SCOPED mask layers rather than one blob: a static bad-pixel mask (excluded everywhere), a peak mask (only suppresses peak finding), a per-shot saturation threshold, a hot/persistently-bright mask refreshed from running frame buffers, and a resolution-annulus mask tied to detector position. Also the median-of-recent-frames background and local-median peak screening. Paper — papers/masking/01_Barty_2014_Cheetah_JApplCryst.pdf.
category: masking
role: mask architecture (literature method — layer taxonomy and scoping)
gate: always applicable as the organizing principle for a run's mask layers
status: not-wired (the layer discipline is followed; per-shot/dynamic layers are not)
---

# Masking · 10 — Cheetah mask layers (Barty et al. 2014)

The reference practice for **how a mask set is organized** at an XFEL, from the software
that processed the first serial-femtosecond-crystallography experiments. Its lesson is
scoping, not statistics: different masks must act at different stages.

## Principle

One "mask" is really several, each with a **different scope**, and conflating them is a
data-analysis bug:

| Layer | Scope — where it acts |
|---|---|
| **Bad-pixel mask** (user/static binary image) | excluded at *every* stage of processing |
| **Peak mask** | ignored **only** during peak finding — blocks regions that produce spurious peaks or speed up the search, without discarding their data |
| **Saturation threshold** | per-shot: pixels above a detector-specific intensity level are flagged on that frame only |
| **Hot-pixel mask** | pixels persistently above threshold over a buffer of recent frames; periodically recalculated during the run |
| **Halo / persistently-bright mask** | pixels frequently in the top intensity fraction — beam halo, scatter |
| **Resolution mask** | an annulus about the direct beam from a user resolution range; regenerated when the detector moves |

Two supporting operations that shape what needs masking at all:

- **Running background**: pixel-wise **median** through a buffer of recent non-hit frames.
  The paper is explicit that the median beats an average here — averaging is dragged by the
  same outlier pixels a mask is meant to catch.
- **Local-median peak screening**: estimate a pixel's local background as the median over a
  (2r+1)² box, with the box area at least twice any plausible Bragg peak, so the median stays
  "blind" to the peak it is judging.

## Parameters

| Param | Guidance from the paper |
|---|---|
| saturation level | detector-dependent, user-specified — not a universal constant |
| frame buffer depth | enough recent non-hit frames for a stable pixel-wise median |
| hot-pixel recalculation period | periodic during the run, not once at the start |
| local-median box `r` | box area ≥ 2× the area of any potential Bragg peak |
| resolution annulus | from the requested resolution range; **must** be regenerated on detector motion |

## Decision rules

1. Ask of every candidate mask: *at which stage should this act?* If the answer is "only at
   peak finding", it is a peak mask, not a bad-pixel mask — putting it in the bad-pixel mask
   silently deletes usable data.
2. Static defect masks come from calibration ([09](09_rmm_robust_mask_maker.md)); per-shot
   conditions (saturation) and slow drifts (hot pixels) need their own, dynamic layers.
3. Any geometry-derived mask (resolution annulus, beamstop) is invalidated by detector
   motion — tie it to the geometry, not to the run.
4. Keep the layers separate and countable; report each layer's pixel count.

## Evidence (Run0475)

This repo follows the layer discipline — the mask agent writes disjoint layers with
per-layer counts (`mask_rationale.md`, `_mask_layer_counts.json`; trial_07: 6 layers,
5.81 % of panel area) and keeps geometry layers ([01](01_geometry_gap.md), beamstop) apart
from defect layers. **Not yet adopted**: the per-shot saturation flag and the
buffer-refreshed hot-pixel/halo layers — this pipeline masks on the *summed* image, so a
pixel that saturates on a few shots is currently invisible. Closest analogue here is the
across-shot statistical family ([07](07_fano_dispersion_mask.md),
[08](08_js_shape_mask.md)), which judges each pixel's time series instead of a running
buffer.

## When to use

As the checklist when designing or reviewing a run's mask set, and whenever a mask is about
to be widened "just to be safe" — that instinct usually means a narrower-scoped layer is
what is actually wanted.

## Trade-offs

Dynamic layers need frame-by-frame access and buffer memory. A peak mask that is too
generous hides real hits; a resolution mask left stale after a detector move is worse than
none, because it looks deliberate.

## Outputs

A layered mask set with per-layer provenance and counts. Paper:
`papers/masking/01_Barty_2014_Cheetah_JApplCryst.pdf` (doi:10.1107/S1600576714007626,
open access).

## Links

Part of: [masking](README.md). Defect layers from [09](09_rmm_robust_mask_maker.md);
per-layer accounting is golden rule 2 in the [category README](README.md); the across-shot
analogue of its dynamic layers is [07](07_fano_dispersion_mask.md) /
[08](08_js_shape_mask.md).
