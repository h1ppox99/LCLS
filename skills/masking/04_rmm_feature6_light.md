---
name: xray-masking-rmm-feature6-light
description: RMM Feature-6 mask — catches pixels that pass dark tests (F2/F3) but are persistently high/low under illumination, via a local robust plane fit and per-pixel SNR. Use when you suspect illumination-dependent bad pixels. Note the faithful version needs per-frame images (psana); the local approximation here uses the run-average image.
category: masking
role: signal-independent
gate: suspected illumination-dependent bad pixels that dark masks miss
status: wired
---

# Masking · 04 — RMM Feature-6 mask (robust statistics under illumination)

## Principle

From Sadri et al. (2022). Some pixels look fine in the dark (F2/F3 pass) but are persistently
high/low **when illuminated**. For each pixel, robustly fit a plane to a local window, take the
pixel's SNR vs that local background, average over frames, then run outlier detection on that
map (`|SNR| > λ` ⇒ bad).

## Parameters

| Param | Value | Why |
|---|---|---|
| `window` | 15×15 | Paper-recommended local window (~225 px, robust). |
| `numModelParams` | 4 | Local background = plane / low-order surface. |
| `MSSE_LAMBDA` | 3.0 | Inlier cutoff for the local plane fit. |
| `asic_block` | 256 | Grouping unit for the outlier-detection stage (Jungfrau ASIC). |
| `lambda_SNR` (λ) | 8.0 | Final flag threshold, consistent with F2/F3. |

## Evidence (Run0475)

1 635 / 1 048 576 ≈ **0.156 %** (persistently bright 668, dark/dead 967).

## When to use

You suspect illumination-dependent bad pixels (flickers, response nonlinearities) that dark
masks ([method 3](03_rmm_dark_based.md)) miss.

## Trade-offs

The faithful version needs **per-frame 2D images** (raw XTC via `psana`, Linux-only). This
machine's approximation runs the *identical* math on the whole-run **average** image, so it
catches pixels *persistently* anomalous vs their neighborhood but **not** intermittent
flickers. State this limitation when reporting.

## Outputs

Feature-6 mask layer (persistently bright ∪ dark/dead). Provenance:
`masking_RMM_feature6_light/` — `build_feature6.py`, `feature6.npz`, `README_Feature6说明.md`.

## Links

Part of: [masking](README.md). Dark-based sibling: [03](03_rmm_dark_based.md).
