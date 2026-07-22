---
name: xray-masking-geometry-gap
description: Geometry gap mask in assembled-frame coordinates, derived purely from ix/iy pixel-coordinate maps — flags inter-module gaps and the outer border where no physical pixel exists. Use whenever working in assembled coordinates so empty cells are not read as zero-intensity real pixels.
---

# Masking · Method 1 — Geometry gap mask (assembled frame)

**Family:** signal-independent. **Part of:** [masking](README.md).

## Principle

In the *assembled* detector image there are cells where **no physical pixel exists** —
inter-module gaps and the outer border. Derive the mask purely from the `ix/iy`
pixel-coordinate maps (where each detector pixel lands in the assembled grid); a cell with no
pixel mapped to it is a gap. No intensity or calibration statistics involved.

## Parameters

None tuned — it is geometry. Output is in assembled coords `(1030, 1064)`, `True = gap/border`.

## Result (Run0475)

47 344 / 1 095 920 cells ≈ **4.32 %** are gaps.

## When to use

Any time you work in assembled coordinates (plotting, 2D correlation, azimuthal regridding).
Prevents "empty" cells from being read as zero-intensity real pixels.

## Trade-offs

Only covers non-existent pixels; says nothing about *bad* real pixels. Complementary to
methods [00](00_status_baseline.md)/[02](02_pyfai_azimuthal_sigmaclip.md)/[03](03_rmm_dark_based.md),
never a replacement.

## Repo

`masking_gap_geometry/` — `gap_mask.npy`, `gap_geometry.npz`, `parameters.json`.
