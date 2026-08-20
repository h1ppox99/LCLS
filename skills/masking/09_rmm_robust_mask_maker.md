---
name: xray-masking-rmm-robust-mask-maker
description: Literature method (Sadri et al. 2022, J. Appl. Cryst. 55, 1549) — Robust Mask Maker (RMM) builds bad-pixel masks by modelling the NORMAL pixels with a robust Gaussian fit (MSSE scale estimator on inliers only) per ASIC block, then flagging |SNR| > lambda outliers across several features (pixel offset, dark STD, illuminated response). Signal-independent, so it removes detector defects without eating Bragg signal; its stated purpose is stopping peak finders from labelling bad pixels as Bragg peaks. Paper — papers/masking/05_Sadri_2022_Robust_Mask_Maker_JApplCryst.pdf.
category: masking
role: signal-independent (literature method — the source of methods 03/04)
gate: calibration constants (dark, illuminated run) available; the recommended default defect mask
status: wired (as methods 03 and 04)
---

# Masking · 09 — Robust Mask Maker (Sadri et al. 2022)

Literature source for this repo's [03_rmm_dark_based](03_rmm_dark_based.md) (features F2/F3)
and [04_rmm_feature6_light](04_rmm_feature6_light.md) (feature F6). This file records what
the paper actually prescribes so the parameters here can be traced back to their origin.

## Principle

Do not model bad pixels — **model the normal ones robustly and call everything else an
outlier**. For a per-pixel feature `x(i)`, fit a robust Gaussian: the robust mean μ_R and the
scale σ_R via **MSSE** (modified selective statistical estimator), which estimates the noise
scale from *inliers only*, so a population of bad pixels cannot inflate it the way a plain
standard deviation would. Then

```
SNR(i) = (x(i) − μ_R) / σ_R        flag if |SNR| > λ
```

The fit is done **per ASIC block** (a detector's natural fabrication unit), because offset
and noise levels differ block to block; a global fit would mask whole healthy blocks and miss
locally-bad pixels. Several features are computed independently and the resulting masks are
unioned — the paper's point is that different defects only show up in different features.

| Feature | Input | Catches |
|---|---|---|
| F2 — pixel offset | pedestal / dark mean | anomalous bias, dead pixels |
| F3 — pixel STD | dark rms | noisy/flickering (high) and dead (near-zero) pixels |
| F6 — response under illumination | local robust plane fit on illuminated frames | pixels that pass the dark tests but are persistently high/low when lit |

## Parameters

Paper defaults, carried into this repo's methods 03/04:

| Param | Value | Why |
|---|---|---|
| `lambda_SNR` (λ) | 8.0 | paper's `darkSNR`; conservative — only true outliers |
| `MSSE_LAMBDA` | 4.0 (dark), 3.0 (F6 plane fit) | inlier cutoff *inside* the robust fit |
| `asic_block` | detector ASIC size (256 here) | the local unit for offset/noise comparison |
| F6 window | 15×15, 4 model params | local background = plane / low-order surface |

## Decision rules

- Run it whenever dark calibration exists — this is the safe default defect mask.
- Union the features; do **not** substitute one for another (a pixel bad only under
  illumination is invisible to F2/F3, and vice versa).
- Judge per block. A whole-detector fit is the documented failure mode.

## Evidence (Run0475)

Reproduced in this repo: F2∪F3 flagged 2 363 / 1 048 576 px (**0.225 %**); F6 flagged
1 635 px (**0.156 %**). Of the RMM flags, 1 661 px overlap the signal-dependent
[pyFAI sigma-clip mask](02_pyfai_azimuthal_sigmaclip.md) (a genuinely bad ASIC band), while
702 RMM-only px are calibration-bad but invisible in the average image — the paper's central
claim, confirmed. Conversely ~8 035 pyFAI-only px are real Bragg/texture signal, i.e. what a
signal-dependent mask costs you.

## When to use

Always, as the defect floor. Especially before any automated peak finding: the paper's
motivating figure is a peak finder marking bad pixels as Bragg peaks on a *blank* frame,
which then triggers the hit-finder to store uninformative data.

## Trade-offs

Needs calibration inputs; static between calibrations. The faithful F6 needs per-frame
images (this repo approximates it on the run-average — states that limitation in
[04](04_rmm_feature6_light.md)). Robust fitting costs more than a threshold, but runs once
per calibration, not per frame.

## Outputs

Per-feature masks + their union, with per-feature pixel counts logged. Paper:
`papers/masking/05_Sadri_2022_Robust_Mask_Maker_JApplCryst.pdf`
(doi:10.1107/S1600576722009815, open access).

## Links

Part of: [masking](README.md). Implemented here as [03](03_rmm_dark_based.md) (F2/F3) and
[04](04_rmm_feature6_light.md) (F6). Same authors' library paper:
[12_rgflib_robust_statistics](12_rgflib_robust_statistics.md).
