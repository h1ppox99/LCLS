---
name: xray-masking-rgflib-robust-statistics
description: Literature method (Hadian-Jazi & Sadri 2023, Acta Cryst. D79, 820) — RGFlib, the Python/C robust-statistics library underneath RMM. Its framing is the load-bearing idea for masking: serial-crystallography data is a MIXTURE of two distributions (true background vs outliers = Bragg peaks or bad pixels), so a robust Gaussian fit that estimates the noise scale from inliers only can label the outliers without ever modelling them. Supplies both robust peak finding and automated bad-pixel-mask making from the same primitives. Paper — papers/masking/06_Hadian_Jazi_Sadri_2023_Robust_Statistics_Python.pdf.
category: masking
role: statistical foundation (literature method — the estimator behind 03/04/09)
gate: whenever a mask threshold is being set on data that may itself contain the defects
status: not-wired (the estimators are reimplemented in 03/04; the library is not a dependency)
---

# Masking · 12 — RGFlib robust statistics (Hadian-Jazi & Sadri 2023)

The statistical foundation the RMM masks ([09](09_rmm_robust_mask_maker.md),
[03](03_rmm_dark_based.md), [04](04_rmm_feature6_light.md)) stand on, published as a
reusable library. Read this one when deciding **how a threshold should be estimated**, not
which pixels to mask.

## Principle

X-ray serial-crystallography data is a **mixture of two probability distributions**: true
data points (background intensities) and outliers (Bragg peaks, or bad pixels on the
detector). Non-robust statistics — plain mean and standard deviation — are valid only when
the data contain a *single* structure with no outliers. Applied to a mixture they are
dragged by the very population you are trying to isolate, so the threshold derived from them
is contaminated by the defects it is supposed to catch. This is the circularity that robust
statistics breaks:

```
fit a single-parameter model (the Gaussian mean) to the data
estimate the noise scale from INLIERS ONLY  (MSSE)
label everything beyond λ·scale as outlier
```

The median is the familiar robust statistic, but the paper notes it is a biased estimator
with low efficiency — hence the scale estimator, not just a median, does the work.

The same primitives serve two tasks the paper presents together:

1. **Robust peak finding** — outliers above a robustly-estimated background are Bragg peaks;
2. **Automated bad-pixel-mask making** — the same outlier labelling applied across
   calibration features gives the mask.

That symmetry is the reason masking and peak finding must be done in the right order: an
unmasked bad pixel is indistinguishable from a peak to the same estimator.

## Parameters

| Param | Meaning |
|---|---|
| MSSE inlier cutoff | which residuals count as inliers while estimating the scale (paper defaults 3–4) |
| λ (SNR threshold) | how many robust σ before a point is called an outlier (8 for dark masks) |
| structure count | the method assumes ONE inlier structure per fitted unit — hence per-ASIC-block fitting upstream |

## Decision rules

- Never derive a mask threshold from a non-robust mean/σ of data that contains the defects.
  If a plain σ must be used, the outliers have to be rejected first — which is the problem
  you started with.
- Fit within units that contain a single structure (ASIC block, q-ring, λ-stratum). A fitted
  unit spanning two different populations violates the model's assumption.
- Robust ≠ conservative: state λ, and report how many points were labelled outliers.

## Evidence (Run0475)

Reimplemented rather than imported here (`RobustGaussianFittingLibrary`; on Apple Silicon
the shipped `RGFLib.so` is a Windows DLL and the C core must be recompiled — noted in
[03](03_rmm_dark_based.md)). The same "estimate the null from the population itself"
discipline is used throughout this skill set beyond RMM: the empirical stratum fences in
[qa/08 pixel photon statistics](../qa/methods/08_pixel_photon_statistics.md), the robust
MAD-based z-scores in [05](05_azimuthal_residual.md) and [06](06_azimuthal_sector_itheta.md),
and the MAD noise floor in the QA glitch lens.

## When to use

Whenever setting a fence on a population that may contain the anomalies being hunted —
i.e. essentially every masking decision made from data rather than from geometry.

## Trade-offs

Robust fits cost more than moments and need an inlier-cutoff parameter of their own (one
more knob to state). They assume a single inlier structure per fitted unit; a bimodal
inlier population (two gain regimes in one block) breaks that assumption silently.

## Outputs

Robust μ, robust σ, and an outlier label per point — feeding masks and peak lists alike.
Paper: `papers/masking/06_Hadian_Jazi_Sadri_2023_Robust_Statistics_Python.pdf`
(doi:10.1107/S2059798323005855, open access).

## Links

Part of: [masking](README.md). Estimator behind [09](09_rmm_robust_mask_maker.md) →
[03](03_rmm_dark_based.md) / [04](04_rmm_feature6_light.md). Same
population-as-its-own-null logic:
[qa/08](../qa/methods/08_pixel_photon_statistics.md).
