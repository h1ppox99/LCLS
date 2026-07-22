# Mask Rationale — agent_trial_03

**Run:** 475 (Jungfrau1M, LaB6)
**Image:** `sum_assembled.npy` (1064 x 1030, float64) — calibrated, normalized, assembled sum of 2747 kept shots.
**Mask:** `mask_assembled.npy` (1064 x 1030, bool, True = masked).
**Total masked:** 114,223 px (10.42% of image; **6.38% of panel area**).
**Unmasked:** 981,697 panel pixels available for integration.

---

## Layer-by-layer breakdown

### Layer 0 — Zero/dead pixels + gap-edge dilation

| Metric | Value |
|---|---|
| New pixels masked | 75,654 |
| % of image | 6.90% |

**What:** Union of (a) exactly-zero pixels dilated by 1 px and (b) all negative pixels.

**Why:**
- **47,369 exactly-zero pixels** are geometry holes — rows where no physical detector pixel exists in the assembled grid: inter-module gap rows 256–257, 806–807, and the large inter-panel gap rows 514–549, plus gap columns 256–257, 514–515, 772–773. These carry no signal and read as zero intensity, which would bias any azimuthal average or integration downward.
- **1-px binary dilation** (3x3 square structuring element, 1 iteration) expands the gap coverage by 12,433 pixels to mask unreliable gap-edge pixels. Pixels adjacent to a geometry gap can have partial coverage (the assembled pixel may overlap the gap boundary) or alignment errors from the pixel-coordinate mapping, producing systematically low readings.
- **15,937 negative pixels** are included directly (no dilation). Negative values in a summed calibrated image indicate calibration failures (pedestal over-subtraction), dead pixels whose dark-subtracted output fluctuates around zero, or extreme hot-pixel artifacts from gain-switching errors. The most extreme cases (547 pixels with |value| > 10^6, all negative, clustered near stuck gain-switch pixels) are unmistakable hardware defects. Moderately negative pixels in the beamstop shadow (~200 per row in rows 977–1001) arise from background subtraction in the beamstop-occluded region, where the true signal is near zero and noise drives values below zero.

### Layer 1 — Known-bad pixels (calibration status)

| Metric | Value |
|---|---|
| New pixels masked | 2,593 |
| Raw (pre-overlap) | 4,565 |
| % of image | 0.24% |

**What:** `calib/status_bad.npy` (2, 512, 1024), a boolean map of factory-flagged bad pixels collapsed across gain stages, mapped from raw detector coordinates to assembled space via `calib/ix.npy` and `calib/iy.npy`.

**Why:** The pixel-status array from the Jungfrau calibration store identifies pixels known to be defective from factory testing or calibration updates. These include pixels with anomalous gain, unstable pedestals, or non-responding channels. Of the 4,565 raw bad pixels, 1,972 overlap with already-masked zero/negative pixels (dead pixels that are both status-flagged and zero-valued). The remaining 2,593 are pixels that appear numerically normal in the sum but are documented hardware defects — they may have incorrect gain correction or intermittent behavior that biases the average.

**Reference:** skills/masking/00_status_baseline.md (Method 0 — always-on baseline).

### Layer 2 — Beamstop shadow

| Metric | Value |
|---|---|
| New pixels masked | 31,114 |
| Raw (pre-overlap) | 38,111 |
| % of image | 2.84% |

**What:** Two sub-components:
1. **Horizontal shadow band (rows 970–1006, full width):** The beamstop arm casts a horizontal shadow across the detector, centered on the beam-center row (~992). Masked as full rows.
2. **Direct-beam disc (radius 15 px around row 992, col 35):** Circular mask at the beam-center attachment point.

**Why (2a — shadow band):** Row-by-row non-zero median analysis shows the shadow:
- Normal rows (e.g., row 960): median ~411
- Row 970: median 309 (entering shadow, 77% of normal)
- Row 974: median 190 (47% of normal — below 50% threshold)
- Rows 977–1001: median 80–130 (20–33% of normal — deep shadow core)
- Row 1004: median 213 (53%, exiting)
- Row 1007: median 329 (82%, back to normal)

The shadow extends the full detector width, with deeper suppression at high columns (far from beam center): cols 800–1000 show 84% suppression vs. 43% at cols 0–100. Rows in this band carry attenuated and distorted signal from the beamstop arm; including them in azimuthal integration would systematically depress the I(q) profile at q values corresponding to these rows.

**Why (2b — beam disc):** The direct-beam vicinity (within ~15 px of center) contains extreme outlier pixels (values to -4.2 × 10^8 from gain-switching failures) and residual scatter leaking around the beamstop. A circular mask of 15-px radius (709 pixels) conservatively covers this region.

### Layer 3 — ASIC boundary double-width edges

| Metric | Value |
|---|---|
| New pixels masked | 3,963 |
| Raw (pre-overlap) | 16,688 |
| % of image | 0.36% |

**What:** The outermost row and column of each ASIC block in assembled coordinates. Masked rows: 0, 255, 258, 513, 550, 805, 808, 1063. Masked columns: 0, 255, 258, 513, 516, 771, 774, 1029.

**Why:** The Jungfrau1M's 2 panels each contain 2×4 ASICs (256×256 pixels). In the assembled image, the edge pixels of adjacent ASICs (which share a physical boundary) map to single assembled-grid cells that receive signal from double-width detector area. Median-ratio analysis confirms the bias:

| Edge row | Median | Neighbor median | Ratio |
|---|---|---|---|
| 255 | 661 | 310 | 2.14× |
| 258 | 650 | 297 | 2.19× |
| 513 | 566 | 344 | 1.64× |
| 550 | 590 | 426 | 1.38× |
| 805 | 744 | 405 | 1.84× |
| 808 | 761 | 414 | 1.84× |

These systematically elevated pixels would produce artificial spikes in azimuthal profiles at the corresponding q values. Column boundaries (cols 255, 258, 513, 516, 771, 774) show analogous effects. The large raw count (16,688) reflects the cross-pattern of 8 full rows + 8 full columns; most of those cells were already masked by Layer 0 (gap-adjacent) or Layer 2 (beamstop), leaving 3,963 newly masked.

**Reference:** skills/masking/01_geometry_gap.md (Method 1, complementary to gap-only masking).

### Layer 4 — Azimuthal-residual parasitic blob

| Metric | Value |
|---|---|
| New pixels masked | 899 |
| Raw (pre-overlap) | 899 |
| % of image | 0.08% |

**What:** One diffuse parasitic-scattering blob at centroid (row 686, col 388), radius ~467 px from beam center, azimuth ~-42 degrees. Detected via the azimuthal-residual method (skills/masking/05_azimuthal_residual.md): radial-median model → destriped residual → Gaussian smoothing (sigma=8) → z-score with hysteresis (z_seed=5.0, z_grow=3.0, min_size=150).

**Why:** This blob is a parasitic scatter feature (likely a window/kapton reflection) that breaks azimuthal symmetry. It is **off-ring** (radius 467, no powder ring within 60-px margin) and too diffuse for per-pixel sigma-clipping (FWHM ~33 px, only ~0.7 sigma_pixel per pixel). The method's destriping step is critical: without it, detector column noise suppresses the z-score from 6.2 to 3.8, making the blob invisible to any reasonable threshold.

The algorithm found 11 total components above z_seed=5.0:
- 9 on-ring components (Bragg spots from large LaB6 grains, radii ~845–1245 px) — correctly **skipped** under the off-ring policy to preserve real diffraction signal
- 1 near-beam component (r ≈ 38 px) — correctly **skipped** (beamstop territory, handled by Layer 2)
- 1 off-ring parasitic blob — **kept** (899 px after base-mask subtraction)

**Reference:** skills/masking/05_azimuthal_residual.md (Method 5 — blob detection, validated against agent_trial_02).

---

## Layers considered but not applied

### pyFAI azimuthal sigma-clipping (Method 2)

**Skipped.** This signal-dependent method clips per-pixel outliers vs. the per-q-ring model, catching cosmics and Bragg spots. However:
1. The input is already a sum of 2,747 shots — cosmic rays are diluted by averaging.
2. The LaB6 Bragg spots from large grains are real signal that should be preserved for calibration use.
3. The method would mask ~0.9% additional pixels, most of which are genuine texture/Bragg signal rather than detector defects.

The signal-independent layers (0–1, 3) already cover the detector defects that Method 2 would also catch (overlap of 1,661 pixels per the skill documentation).

### RMM dark-based F2/F3 (Method 3)

**Not independently applied.** The RMM mask (2,363 pixels in raw space) overlaps heavily with Layers 0 and 1. The offset-outlier (F2) and noise-outlier (F3) pixels are largely the same pixels that appear as zero/negative in the sum or are flagged by status_bad. The 702 RMM-only pixels (calibration-bad but not visible in the average) would add marginal coverage at the cost of needing the RobustGaussianFittingLibrary dependency. The benefit is negligible for the assembled sum use case.

### RMM Feature-6 under illumination (Method 4)

**Skipped.** Feature-6 catches pixels that are normal in the dark but persistently anomalous under illumination. The faithful version requires per-frame images via psana (Linux-only). The local approximation (running on the average image) would catch persistently anomalous pixels vs. their local neighborhood, but these are already substantially covered by the status_bad and zero/dead layers. The 1,635 Feature-6 pixels overlap heavily with existing layers. Additionally, the limitation of using the average instead of per-frame data means intermittent flickers would be missed regardless.

---

## Summary table

| Layer | New px | Cumulative px | Cumulative % (image) | Cumulative % (panel) |
|---|---|---|---|---|
| 0. zero/dead + dilation | 75,654 | 75,654 | 6.90% | 3.94% |
| 1. status_bad (calib) | 2,593 | 78,247 | 7.14% | 4.19% |
| 2. beamstop shadow | 31,114 | 109,361 | 9.98% | 5.90% |
| 3. ASIC boundary edges | 3,963 | 113,324 | 10.34% | 6.28% |
| 4. azimuthal blob | 899 | 114,223 | 10.42% | 6.38% |

**Final: 114,223 masked / 1,095,920 total (10.42%), leaving 981,697 unmasked pixels (93.62% of panel area) for downstream integration.**
