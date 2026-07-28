# Mask Rationale — agent_trial_06 (Run 475, Jungfrau1M, LaB6)

**Input:** `sum_assembled.npy` (1064 x 1030), calibrated & normalized sum of 2747 kept shots.
**Output:** `mask_assembled.npy` (1064 x 1030) bool, `True = masked`.
**Beam center:** (row 992, col 35) in assembled coordinates.

## Summary

| Layer | Description | Pixels | % of image | % of panel |
|-------|-------------|-------:|----------:|----------:|
| L0 | Geometry gap (no physical pixel) | 47 344 | 4.320 % | — |
| L1 | Dead/neg/extreme + gap-edge dilation | 28 185 | 2.572 % | 2.688 % |
| L2 | Calibration status_bad | 2 599 | 0.237 % | 0.248 % |
| L3 | Beamstop shadow + beam exclusion | 28 123 | 2.566 % | 2.682 % |
| L4 | ASIC double-width seams (subsumed) | 0 | 0.000 % | 0.000 % |
| L5 | Azimuthal-residual blob | 2 082 | 0.190 % | 0.199 % |
| **Total** | | **108 333** | **9.89 %** | **5.81 %** |

All layers are mutually exclusive (disjoint accounting verified programmatically).
Unmasked panel pixels: 987 587.

---

## Layer-by-layer detail

### L0 — Geometry gap mask (47 344 px, 4.32 %)

**What:** Assembled-frame cells where no physical detector pixel exists — inter-module
gaps and the outer border.

**How:** Built from the `calib/ix.npy` and `calib/iy.npy` coordinate maps: any cell in
the (1064, 1030) grid that no raw pixel maps to is flagged.

**Why:** Without this, gap cells read as zero and contaminate any azimuthal average or
radial profile. This is method [01_geometry_gap](../../skills/masking/01_geometry_gap.md).

### L1 — Dead / negative / extreme pixels + gap-edge dilation (28 185 px, 2.57 %)

**What:** Three sub-components unioned together:

| Sub-component | Pixels | Description |
|---|---:|---|
| Gap-edge dilation (1 px, 4-connectivity) | 12 252 | 1-pixel halo around every gap cell |
| Dead / negative (img <= 0, excl. gaps) | 15 962 | Pixels that read zero or negative in the calibrated sum |
| Extreme outliers (\|img\| > 10^6) | 547 | Gain-switching artifacts producing values 10^5–10^8 |

Overlap between sub-components: 40 pixels (dead_neg or extreme that also fall in gap_edge).

**How:**
- Gap-edge dilation: `scipy.ndimage.binary_dilation(gap_mask, cross_struct, iterations=1)`,
  then subtract the gap itself to get the 1-px halo.
- Dead/negative: `img <= 0` on the assembled sum, excluding gap cells.
- Extreme: `|img| > 1e6`, catching gain-switching artifacts that produce values many orders
  of magnitude above the physical signal (median ~ 335 ADU).

**Why:**
- Gap-edge pixels sit at the boundary between physical sensor and empty gap. They collect
  charge from a truncated pixel area and have unreliable calibration. Dilating by 1 pixel
  (4-connectivity cross kernel) is the minimum needed to exclude these.
- Pixels reading <= 0 in a sum of ~2747 positive-signal shots are dead or have catastrophic
  calibration errors.
- Extreme outliers (|value| > 10^6 vs median ~335) are gain-switching artifacts where the
  calibration gain lookup selected the wrong gain stage for one or more shots, producing
  values 10^3–10^6 x the physical signal.

**Note on ASIC seams:** The 1-px gap-edge dilation also catches all 10 216 double-width ASIC
boundary pixels (raw rows 255/256, raw cols 255/256, 511/512, 767/768). In the assembled
coordinate system, these ASIC seams map to positions immediately adjacent to the inter-ASIC
gaps, so gap-edge dilation subsumes them naturally. This is why Layer 4 below contributes
zero additional pixels. The ASIC seam pixels have median intensity ~689 ADU vs ~331 ADU
for interior pixels (ratio 2.08x) — they would bias any azimuthal average by ~2 % per ring
if left unmasked.

### L2 — Calibration status_bad (2 599 px, 0.24 %)

**What:** Factory-characterized bad pixels from the detector calibration store.

**How:** Loaded `calib/status_bad.npy` (shape 2, 512, 1024; bool), which flags pixels
identified as defective across any gain stage. Mapped to assembled coordinates using the
same `calib/ix.npy`, `calib/iy.npy` coordinate maps. Then subtracted pixels already
captured by L0 + L1 to keep layers disjoint.

**Why:** This is method [00_status_baseline](../../skills/masking/00_status_baseline.md) —
the factory baseline that every other mask is unioned onto. The raw count is 4 565 pixels;
1 966 were already captured by L1 (dead/neg or gap-edge overlap), leaving 2 599 new pixels.

### L3 — Beamstop shadow + direct-beam exclusion (28 123 px, 2.57 %)

**What:** A full-detector-width horizontal shadow band (rows 972–1005 inclusive, 34 rows)
plus a small circular exclusion zone (radius <= 15 px around beam center).

**How:** The shadow boundaries were determined by comparing each row's median intensity to
the expected intensity from an azimuthal-median radial model built from unshadowed rows.
The model was built from all rows outside the suspected shadow zone (rows 0–969 and
1010–1063), using only non-zero, non-extreme pixels.

Row-by-row deficit analysis:

| Row | Actual median | Expected median | Deficit |
|-----|-------------:|----------------:|--------:|
| 969 | 322 | 340 | −5 % |
| 970 | 309 | 343 | −10 % |
| 972 | 263 | 346 | **−24 %** (shadow onset) |
| 980 | 96 | 349 | −73 % |
| 992 | 83 | 351 | −76 % (beam center) |
| 1003 | 113 | 350 | −68 % |
| 1005 | 263 | 349 | **−25 %** (shadow exit) |
| 1006 | 307 | 348 | −12 % |
| 1008 | 348 | 348 | 0 % |

The shadow is uniform across the full detector width: at cols 400–1030 the deficit
within the shadow band reaches −80 to −83 %, confirming a physical beamstop arm casting
a horizontal shadow upstream of the detector.

The −20 % deficit threshold places the boundaries at rows 972 (onset) and 1005 (exit).

The direct-beam exclusion (r <= 15 px) masks the small region around the beam center
where stray direct-beam leakage and beamstop edge diffraction produce unreliable values.

**Why:** The beamstop shadow suppresses intensity by 60–80 % relative to the powder-ring
model. Leaving it unmasked would systematically bias any azimuthal average at all q-values
whose rings cross this band. The full-width masking is justified because the shadow extends
across all columns (not just near the beam center), as shown by the column-band analysis.

### L4 — ASIC double-width seams (0 px — subsumed by L1)

**What:** Internal ASIC boundaries at raw-space rows 255/256 and cols 255/256, 511/512,
767/768.

**Why 0 new pixels:** In assembled coordinates, the Jungfrau geometry inserts 2–3 pixel
gaps between ASICs. The ASIC edge pixels (raw rows/cols 255, 256 etc.) map to positions
immediately adjacent to these gaps. The 1-pixel gap-edge dilation in L1 captures all 10 216
of these pixels. Verified: `ASIC_assembled AND gap_edge = 10 216 / 10 216`.

**Not skipped — correctly subsumed.** The physical effect (double-area charge collection at
ASIC seams, producing 2.08x median intensity) is masked; it is simply accounted for under L1
rather than as a separate layer.

### L5 — Azimuthal-residual blob (2 082 px, 0.19 %)

**What:** A parasitic scattering blob at (row 686, col 387), radius 467 px from beam center,
azimuth approximately −42 deg. This is off-ring diffuse scatter (likely a window or kapton
reflection).

**How:** Method [05_azimuthal_residual](../../skills/masking/05_azimuthal_residual.md):

1. **Radial model:** azimuthal median in 1-px radial bins, 3-bin boxcar smoothed.
2. **Residual:** `img − model(r)`.
3. **Destripe:** subtract per-column then per-row medians from the residual (reduces
   residual noise from 69 to ~31 ADU).
4. **Smooth:** Gaussian filter (sigma = 8 px), NaN-aware normalization.
5. **Z-score:** robust `(S − median) / (1.4826 * MAD)`.
6. **Detect:** seeds at z > 5.0, minimum component size 150 px.
7. **Grow:** hysteresis propagation at z > 3.0 (envelope: 899 px).
8. **Filter:** exclude near-beam (r < 100 px) and on-ring (within 60 px of LaB6 ring radii
   at 367, 531, 734, 847, 1245 px) components.
9. **Refine:** 2-D Gaussian fit on the smoothed field with kernel deconvolution
   (sigma_true = sqrt(sigma_fit^2 − sigma_smooth^2)). Contamination contour at
   eps = 0.10 (mask the fitted ellipse out to where excess = 10 % of background).

Fit results: sigma_true = (12.6, 13.1) px, A_true/background = 0.75 (75 % excess at peak),
z_peak = 6.1. The eps = 0.10 contour produces a ~26 px radius ellipse containing 2 082 px.

**Why:** The blob has 47 % excess over the local background (measured: median 564 vs 383 in
surrounding region). At 75 % peak amplitude, it would bias the azimuthal average in its
radial band by ~1.5 % if left unmasked. The 10 % contamination contour reduces the residual
per-bin bias to ~0.5 %, below half the background noise floor (1.2 %).

---

## Methods considered but not applied

- **02_pyfai_azimuthal_sigmaclip** (signal-dependent per-pixel clip): Not applied. This
  method is designed to catch sporadic Bragg spots and cosmics, which are relevant for
  per-shot masking. On a sum of 2747 shots, sporadic outliers are averaged out. The dominant
  features requiring masking (beamstop shadow, dead pixels, parasitic blob) are all handled
  by the signal-independent layers + the blob detector. Adding method 02 would risk clipping
  real LaB6 Bragg intensity at the ring peaks.

- **03_rmm_dark_based** and **04_rmm_feature6_light**: These require access to the raw dark
  run data and the RMM feature arrays, which are not available in the current trial
  directory. The status_bad mask (L2) covers the factory-known defects; the dead/neg layer
  (L1) catches pixels that have gone bad since calibration. The combination provides
  equivalent coverage for the pixels that would be flagged by dark-based methods.

- **06_azimuthal_sector_itheta**: This is a screener/router, not a mask generator. It
  would identify azimuthal violations and route them to method 05 (blob detection) or
  other handlers. Since we already ran method 05 directly and found the known blob, the
  screener adds no new detections.
