---
name: xray-masking
description: Masking skill category for area-detector scattering (Jungfrau1M). Each method that decides which pixels to exclude before integration lives in its own file in this folder. Three families - signal-independent (geometry/dark), signal-dependent (pattern outliers), and statistical across-shot screens (07 Fano dispersion, 08 JS shape) that test each pixel's photon time series against the conditional Poisson null. Use when a raw or averaged frame has bad columns, hot/dead/stuck/flickering pixels, panel gaps, cosmic rays, Bragg spots, or spectral contamination.
category: masking
role: category-index
status: wired
---

# Masking (category index)

**Goal.** Produce a boolean mask `(2, 512, 1024)`, `True = exclude`, marking pixels that must
not enter any azimuthal average, sum, or fit. Masking is *spatial* — it removes pixels, never
whole shots (that is [selection](../selection/README.md)) and never rescales (that is
[normalization](../normalization/README.md)).

## Methods

| File | Method | Family | Masked % (Run0475) |
|---|---|---|---|
| [00_status_baseline.md](00_status_baseline.md) | Static status mask from calibration | signal-independent | (varies) |
| [01_geometry_gap.md](01_geometry_gap.md) | Geometry gap mask (assembled frame) | signal-independent | 4.32 % |
| [02_pyfai_azimuthal_sigmaclip.md](02_pyfai_azimuthal_sigmaclip.md) | pyFAI azimuthal sigma-clipping | signal-dependent | 0.925 % |
| [03_rmm_dark_based.md](03_rmm_dark_based.md) | RMM dark-based (F2 offset + F3 STD) | signal-independent | 0.225 % |
| [04_rmm_feature6_light.md](04_rmm_feature6_light.md) | RMM Feature-6 (under illumination) | signal-independent | 0.156 % |
| [05_azimuthal_residual.md](05_azimuthal_residual.md) | Azimuthal-residual blob detection (diffuse anomalies) | signal-dependent | 0.20 % |
| [06_azimuthal_sector_itheta.md](06_azimuthal_sector_itheta.md) | I(θ) sector screening (shape-agnostic, two-signed) | signal-dependent | (screener) |
| [07_fano_dispersion_mask.md](07_fano_dispersion_mask.md) | Fano-factor dispersion screen (across shots; only method catching stuck F < 0.5) | statistical (across-shot) | 0.016 % |
| [08_js_shape_mask.md](08_js_shape_mask.md) | JS-divergence histogram-shape screen (flicker, pair-arrival/harmonic) | statistical (across-shot) | 0.008 % |
| [09_rmm_robust_mask_maker.md](09_rmm_robust_mask_maker.md) | Robust Mask Maker — robust Gaussian + MSSE per ASIC block (Sadri 2022) | literature · signal-independent | source of 03/04 |
| [10_cheetah_mask_layers.md](10_cheetah_mask_layers.md) | Cheetah layer taxonomy — scoped masks (bad-pixel / peak / saturation / hot / resolution) (Barty 2014) | literature · architecture | — |
| [11_azimuthal_sigma_clip_signal_separation.md](11_azimuthal_sigma_clip_signal_separation.md) | Iterative azimuthal sigma-clipping, signed outliers (Kieffer 2025) | literature · signal-dependent | source of 02 |
| [12_rgflib_robust_statistics.md](12_rgflib_robust_statistics.md) | RGFlib — mixture framing; estimate the fence from inliers only (Hadian-Jazi 2023) | literature · foundation | estimator behind 03/04/09 |
| [13_unet_artifact_segmentation.md](13_unet_artifact_segmentation.md) | U-Net learned artifact segmentation (Yanxon 2023) | literature · signal-dependent (learned) | not applied |

Methods **09–13 are literature methods**: one file per paper in
[`papers/masking/`](../../papers/masking/README.md), recording what the published method
prescribes and how (or whether) this repo implements it. Use them to trace a parameter back
to its origin, or to pick a method this pipeline does not yet have.

**Three families; usually union one signal-independent + one signal-dependent,
and add statistical layers when per-shot frames are available:**

- **Signal-independent** (00, 01, 03, 04) — from geometry or dark calibration; identifies
  intrinsic detector defects, never removes real signal. Safe default.
- **Signal-dependent** (02) — from the diffraction pattern; catches azimuthal outliers
  (cosmics, Bragg spots) but will eat real texture/Bragg signal if unguarded.
- **Statistical, across-shot** (07, 08) — from each pixel's photon time series vs the
  conditional Poisson null (needs per-shot frames + `T·λ` power, gated). Judges
  *behavior in time*, orthogonal to both other families; classes routed, not
  blanket-masked (beam-modulated pixels are reported, never masked).

## Choosing

- Remove defects, keep all signal → `00 ∪ 03` (status + RMM-dark).
- If `calib/ped.npy` exists, **always expose its gain-stage-0 preview and run the
  pedestal F2 probe before Mask Step 2**. Inspect spatially coherent residuals as
  components, not only as single pixels: a compact low-contrast dot can be obvious
  in the pedestal while remaining below an extreme-value threshold in the sum.
- Clean an isotropic/amorphous background → add `02` (azimuthal sigma-clip).
- Working in assembled coords → always union `01` (geometry gaps).
- Suspected illumination-dependent bad pixels → add `04`.
- Diffuse parasitic blobs / scatter ghosts (too low-contrast per pixel for `02`) → add `05`
  (azimuthal-residual, off-ring policy) on the assembled sum.
- Screening for ANY azimuthal-symmetry violation at fixed r (arcs, streaks, shadows,
  negative block deficits) → run `06` (I(θ) sector scan) and route each detection per its
  classification table; blob-like positives get their footprint from `05`.
- Per-shot frames available and a statistically-vetted pixel set wanted (weighted sums,
  calibration products) → run `07` (cheap dispersion screen; the only layer that catches
  stuck pixels, F < 0.5). Suspected gain flicker or spectral contamination, or `07`
  suspects needing classification → add `08` (histogram shape; caught the 2nd-harmonic
  pixel `07` missed by a hair). Both need the full screened shot set — narrow bands have
  no statistical power (measured: 0.57 vs ~5 expected pair events).

## Golden rules

1. Apply as `clean = np.where(mask, np.nan, img)` — never zero-fill masked pixels.
2. **Always log** masked-pixel count + % per layer; keep layers disjoint for exact accounting.
3. A mask is not a selection ([selection](../selection/README.md)) and never rescales
   ([normalization](../normalization/README.md)).
