---
name: xray-masking-unet-artifact-segmentation
description: Literature method (Yanxon et al. 2023, arXiv:2310.16186, APS/ALS) — learned masking - a tunable U-Net CNN segments artifacts in powder XRD images directly from the image, trained against manually-drawn ground-truth masks and scored by recall against them. Cuts artifact-identification time by >50%, fast enough for on-the-fly masking during an experiment, and excluding the predicted artifacts visibly changes the integrated 1D pattern. Known failure - it can miss artifacts when preferred orientation is also present. Paper — papers/masking/07_Yanxon_2023_UNet_XRD_Segmentation.pdf.
category: masking
role: signal-dependent (literature method — learned segmentation)
gate: a labelled mask corpus exists for this detector/beamline AND a human reviews the output
status: not-wired (no labelled corpus for this campaign)
---

# Masking · 13 — U-Net artifact segmentation (Yanxon et al. 2023)

The learned alternative to every hand-tuned rule in this folder: instead of a statistic plus
a threshold, a CNN is trained on human-drawn masks and predicts the artifact map directly.
Included here as the option to reach for when artifacts are *visually* obvious but
statistically awkward.

## Principle

Treat masking as **image segmentation**. A tunable U-Net (encoder–decoder CNN with skip
connections) takes the 2D XRD image and outputs a per-pixel artifact probability; thresholding
it gives the mask. Training targets are manually-implemented ground-truth masks, and the
reported metric is **recall** (true-positive rate) against them — the paper's stated priority
is not missing artifacts, since a missed artifact contaminates the integrated pattern while a
slightly over-drawn mask mostly costs statistics.

Why it can work where rules struggle: artifacts here (detector shadows, beam-stop structures,
scattering from the sample environment, streaks) are defined by their **shape and context**,
which a convolutional model sees directly, whereas a per-pixel or per-ring statistic has to
infer them from intensity alone.

## Parameters

| Param | Guidance |
|---|---|
| architecture depth / width | "tunable" U-Net — trade capacity against inference latency for the on-the-fly case |
| probability threshold | sets the recall/precision balance; the paper optimizes for recall |
| training corpus | manually masked images from the *same* detector and geometry |
| evaluation metric | recall vs ground truth, plus the effect on the integrated 1D pattern |

## Decision rules

1. **Only with a labelled corpus from this detector and geometry.** A model trained
   elsewhere carries that beamline's artifacts, not yours — this is the same
   geometry-boundness that makes validated pixel anchors non-transferable
   ([qa/09](../qa/methods/09_known_material_expectation.md)).
2. **Report the effect on I(q), not just the pixel metric.** The paper's own success
   criterion is that excluding predicted artifacts produces *major changes* in the
   integrated 1D pattern — so the mask must be judged there.
3. **Human in the loop.** A learned mask has no auditable rationale per pixel; pair it with
   a rule-based layer that does, and have a person review the segmentation before it gates
   science.
4. **Known failure mode, stated by the authors**: the algorithm can fail to differentiate
   artifacts when other characteristics such as **preferred orientation** are present. That
   is precisely the regime where texture arcs look like artifacts — the same trap that makes
   [02](02_pyfai_azimuthal_sigmaclip.md) eat real texture.

## Evidence (Run0475)

**Not applied** — no labelled mask corpus exists for this campaign, and one Jungfrau1M
geometry with a corner beam center is a thin training set. Recorded here as the documented
option should a corpus be built. The paper's own numbers: consistent recall against manual
masks, >50 % reduction in artifact-identification time, and inference fast enough for
on-the-fly masking during data collection. This repo's closest hand-built analogues are the
shape-aware layers [05](05_azimuthal_residual.md) (matched-filter blob envelope) and
[06](06_azimuthal_sector_itheta.md) (shape-agnostic sector screening), which achieve
context-sensitivity through explicit models instead of learned ones.

## When to use

Repetitive campaigns on one instrument where artifacts recur and a human has already masked
many frames; on-the-fly masking during collection; artifacts with characteristic shapes that
per-pixel statistics keep missing.

## Trade-offs

Needs labelled data and training infrastructure; the mask is not explainable per pixel;
generalizes poorly across detectors/geometries; degrades exactly where texture is present.
No calibration-independent guarantee — unlike the signal-independent family, it *can*
remove real signal, so it belongs with the signal-dependent methods and their cautions.

## Do not

- Do not ship a learned mask without a rule-based cross-check and a human review.
- Do not reuse a model across a geometry change without re-validation.

## Outputs

Per-pixel artifact probability map → thresholded mask layer, plus the before/after
integrated 1D comparison that justifies it. Paper:
`papers/masking/07_Yanxon_2023_UNet_XRD_Segmentation.pdf` (arXiv:2310.16186, open access).

## Links

Part of: [masking](README.md). Rule-based, shape-aware alternatives:
[05](05_azimuthal_residual.md) and [06](06_azimuthal_sector_itheta.md). Statistical
alternative with auditable thresholds: [09](09_rmm_robust_mask_maker.md) /
[12](12_rgflib_robust_statistics.md).
