---
name: xray-masking-rmm-dark-based
description: RMM (Robust Mask Maker, Sadri 2022) dark-based mask — robustly models normal pixels per ASIC block from calibration constants and flags offset (Feature 2) and noise (Feature 3) outliers. Signal-independent, so it removes detector defects while preserving all diffraction signal. Preferred default defect mask.
---

# Masking · Method 3 — RMM dark-based mask (robust statistics on calibration)

**Family:** signal-independent. **Part of:** [masking](README.md).

## Principle

From Sadri et al., *J. Appl. Cryst.* **55**, 1549 (2022). Don't model bad pixels — robustly
model *normal* pixels. For a feature `x(i)`, fit a robust Gaussian (robust mean μ_R, robust
scale σ_R via the MSSE estimator, which uses only inliers so bad pixels don't pollute the fit),
then `SNR(i) = (x(i) − μ_R)/σ_R`; flag `|SNR| > λ`.

Applied per **ASIC block (256×256)** so offset/noise are compared locally. Two features from
the calibration constants:

| Feature | Input | Catches |
|---|---|---|
| **F2 — pixel offset** | pedestal (`ped[gain0]`) | anomalous bias; dead pixels (ped=0) |
| **F3 — pixel STD** | dark rms (`rms[gain0]`) | noisy/flickering (high) or dead (near-zero) pixels |

Final mask = offset-outliers ∪ std-outliers.

## Parameters

| Param | Value | Why |
|---|---|---|
| `lambda_SNR` (λ) | 8.0 | Paper default `darkSNR=8`; conservative, only true outliers. Smaller ⇒ more aggressive. |
| `MSSE_LAMBDA` | 4.0 | Inlier cutoff *inside* the robust fit (paper default). |
| `asic_block` | 256 | Jungfrau ASIC size — the natural local unit for offset/noise. |
| `gain_stage` | 0 (high gain) | Stage actually used in this low-flux run; union across stages for completeness. |

## Result (Run0475)

2 363 / 1 048 576 ≈ **0.225 %** (offset-high 1231, offset-low 1, std-high 565, std-low 566).
Overlaps the [pyFAI mask](02_pyfai_azimuthal_sigmaclip.md) on 1661 pixels (the "smoking-gun"
bad ASIC band); 702 RMM-only pixels are calibration-bad but invisible in the average image;
the ~8035 pyFAI-only pixels are real Bragg/texture signal, not defects — so to remove defects
and keep signal, prefer this mask.

## When to use

You want to remove **detector defects while preserving all diffraction signal** — this is
signal-independent, so it never eats Bragg peaks. Preferred default defect mask.

## Trade-offs

Needs dark calibration inputs; static between calibrations; won't catch pixels that only
misbehave under illumination (→ [Feature-6](04_rmm_feature6_light.md)).

## Repo

`masking_RMM_darkbased/` — `build_rmm_mask.py`, `rmm_mask.npz`, `README_RMM方法说明.md`.
Install: `pip install RobustGaussianFittingLibrary`. On Apple Silicon the bundled `RGFLib.so`
is a Windows DLL — recompile the C core once:
`cc -O3 -fPIC -shared -o RGFLib.so RGFLib.c -lm`.
