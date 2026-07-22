---
name: xray-masking
description: Masking skill category for area-detector scattering (Jungfrau1M). Each method that decides which pixels to exclude before integration lives in its own file in this folder. Use when a raw or averaged frame has bad columns, hot/dead pixels, panel gaps, cosmic rays, or Bragg spots contaminating an isotropic pattern.
---

# Masking (category index)

**Goal.** Produce a boolean mask `(2, 512, 1024)`, `True = exclude`, marking pixels that must
not enter any azimuthal average, sum, or fit. Masking is *spatial* — it removes pixels, never
whole shots (that is [selection](../selection/README.md)) and never rescales (that is
[normalization](../normalization/README.md)).

Each method is a **separate file** in this folder:

| File | Method | Family | Masked % (Run0475) |
|---|---|---|---|
| [00_status_baseline.md](00_status_baseline.md) | Static status mask from calibration | signal-independent | (varies) |
| [01_geometry_gap.md](01_geometry_gap.md) | Geometry gap mask (assembled frame) | signal-independent | 4.32 % |
| [02_pyfai_azimuthal_sigmaclip.md](02_pyfai_azimuthal_sigmaclip.md) | pyFAI azimuthal sigma-clipping | signal-dependent | 0.925 % |
| [03_rmm_dark_based.md](03_rmm_dark_based.md) | RMM dark-based (F2 offset + F3 STD) | signal-independent | 0.225 % |
| [04_rmm_feature6_light.md](04_rmm_feature6_light.md) | RMM Feature-6 (under illumination) | signal-independent | 0.156 % |
| [05_azimuthal_residual.md](05_azimuthal_residual.md) | Azimuthal-residual blob detection (diffuse anomalies) | signal-dependent | 0.20 % |
| [06_azimuthal_sector_itheta.md](06_azimuthal_sector_itheta.md) | I(θ) sector screening (shape-agnostic, two-signed) | signal-dependent | (screener) |

**Two families, usually union one from each:**

- **Signal-independent** (00, 01, 03, 04) — from geometry or dark calibration; identifies
  intrinsic detector defects, never removes real signal. Safe default.
- **Signal-dependent** (02) — from the diffraction pattern; catches azimuthal outliers
  (cosmics, Bragg spots) but will eat real texture/Bragg signal if unguarded.

**Choosing:**
- Remove defects, keep all signal → `00 ∪ 03` (status + RMM-dark).
- Clean an isotropic/amorphous background → add `02` (azimuthal sigma-clip).
- Working in assembled coords → always union `01` (geometry gaps).
- Suspected illumination-dependent bad pixels → add `04`.
- Diffuse parasitic blobs / scatter ghosts (too low-contrast per pixel for `02`) → add `05`
  (azimuthal-residual, off-ring policy) on the assembled sum.
- Screening for ANY azimuthal-symmetry violation at fixed r (arcs, streaks, shadows,
  negative block deficits) → run `06` (I(θ) sector scan) and route each detection per its
  classification table; blob-like positives get their footprint from `05`.

Apply as `clean = np.where(mask, np.nan, img)` and **always log** masked-pixel count + %.
