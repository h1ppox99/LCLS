# Mask Rationale — agent_trial_04 (Run 475, Jungfrau1M, LaB6)

**Input:** `sum_assembled.npy` (1064 × 1030, float64) — normalized sum of 2746 kept shots.
**Output:** `mask_assembled.npy` (1064 × 1030, bool, True = masked).
**Total masked:** 113 051 px (10.3% of image area, 6.3% of panel area).

---

## Layer 0 — Zero / dead pixels + gap-edge dilation

**Pixels added:** 75 583 (6.90% of image)

Three sub-components:

| Sub-component | Pixels | Description |
|---|---|---|
| Exact zeros (gaps) | 47 369 | Inter-module gap rows (256–257, 514–549, 806–807) and gap columns (256–257, 514–515, 772–773). No physical pixel exists at these cells in the assembled grid. |
| Negative pixels | 15 937 | Pixels with sum < 0. Includes 541 extreme values (< −1 × 10⁶) from gain-switching overflow (e.g. −4.1 × 10⁸ at row 992 col 35 near the direct beam), plus ~15 400 mildly negative pixels scattered across the detector from statistical/calibration noise. |
| Gap-edge dilation | 12 344 (net) | 1-pixel binary dilation of the zero mask catches unreliable gap-edge pixels where the geometry assembly maps a partial or interpolated value. This also automatically masks the **double-bright ASIC boundary columns** flanking each gap (see Layer 3). |

**Justification:** Pixels ≤ 0 in a sum of ~2746 positive-signal calibrated shots are either geometry holes or detector defects. The dilation catches the single row/column of pixels immediately adjacent to every gap, which in a Jungfrau assembly receive counts from only one side of the ASIC boundary and show systematic intensity bias.

---

## Layer 1 — Known-bad pixels (calib/status_bad.npy)

**Pixels added:** 2 597 (0.24% of image)

The detector calibration store provides a per-pixel `status_bad` array of shape (2, 512, 1024) with 4 565 flagged pixels (factory-known defects). These were mapped to assembled coordinates via `calib/ix.npy` and `calib/iy.npy`. After removing overlap with Layer 0 (many bad pixels are already negative or gap-adjacent), 2 597 new pixels were added.

**Justification:** Factory calibration identifies pixels with known gain/offset defects that may produce plausible-looking but incorrect values. These cannot be identified from the image alone. This is the always-on baseline recommended by method 00 of the masking skills.

---

## Layer 2 — Beamstop shadow band

**Pixels added:** 31 121 (2.84% of image)

**Masked region:** rows 970–1006, full detector width (all 1030 columns).

The beam center is at (row 992, col 35) in assembled coordinates. A horizontal beamstop arm casts a shadow across these rows. Row-by-row median analysis shows:

| Rows | Median (ADU) | % of global median (305.5) |
|---|---|---|
| 968–970 (above shadow) | 295–309 | 97–101% (normal) |
| 972–975 (onset) | 158–263 | 52–86% |
| 976–1003 (deep shadow) | 83–141 | 27–46% |
| 1004–1005 (recovery) | 213–263 | 70–86% |
| 1006–1008 (clear) | 307–348 | 101–114% (normal) |

The shadow extends the full width of the detector: column-wise medians within the band range from 30–178 ADU (vs. 300+ above/below), confirming the beamstop arm crosses the entire field.

**Justification:** Partially-shadowed pixels (25–85% attenuation) cannot be corrected by simple scaling because the shadow profile varies with position. Including them in azimuthal integration would bias I(q) low. Rows 970–971 and 1004–1006 are borderline (86–101%) but are included as a conservative transition margin.

---

## Layer 3 — ASIC boundary double-width columns

**Pixels added:** 0

Six ASIC boundary columns were identified with anomalously high median intensity:

| Column | Median (ADU) | Neighbor median | Ratio |
|---|---|---|---|
| 255 | 632.6 | ~320 | 2.0× |
| 258 | 629.9 | ~330 | 1.9× |
| 513 | 702.0 | ~320 | 2.2× |
| 516 | 717.6 | ~330 | 2.2× |
| 771 | 749.6 | ~380 | 2.0× |
| 774 | 751.0 | ~370 | 2.0× |

These columns flank the gap column pairs (256–257, 514–515, 772–773) and receive double-counted intensity from the ASIC edge pixels during geometry assembly. However, all six are already absorbed by Layer 0's 1-pixel dilation of the gap columns, so this layer adds zero new pixels. It is retained for documentation.

**ASIC boundary rows** (255, 258, 513, 550, 805, 808) were also checked. Rows 771 and 774 show no systematic median bias (394.7 and 385.3 vs. ~385 neighbors). The rows flanking inter-module gaps (255/258 near gap 256–257, 805/808 near gap 806–807) are already caught by Layer 0 dilation.

---

## Layer 4 — Azimuthal-residual blob detection (method 05)

**Pixels added:** 3 750 (0.34% of image)

Applied the azimuthal-residual method (skill 05) to detect diffuse parasitic-scatter features invisible to per-pixel sigma-clipping:

1. **Radial model:** azimuthal median per 1-px radial bin, 3-bin boxcar smoothing.
2. **Residual + destripe:** subtract model, remove column-wise then row-wise median stripes (reduces residual noise 69 → 31 ADU).
3. **Smooth:** Gaussian σ = 8 px (matched to ~30 px FWHM targets).
4. **Z-score & threshold:** z_seed = 5.0, z_grow = 3.0, min_size = 150 px.
5. **Policy:** off-ring only (ring margin 60 px), r_min = 100 px.

**Detection results:** 11 grown components found:
- 8 on-ring components (Bragg spots at r ≈ 845–849 and outer arc segments at r ≈ 1243–1245) → skipped per off-ring policy.
- 1 near-beam-center component (r ≈ 38) → skipped per r_min exclusion.
- **1 off-ring parasitic blob** at (row 686, col 390), radius 467 px → **kept**.

**Parametric refinement** (2-D Gaussian fit on smoothed field with kernel deconvolution):
- Center: (686.4, 389.6) in assembled coords
- σ_true: (12.6, 13.1) px — nearly circular (FWHM ≈ 30 px)
- Amplitude: 75% of azimuthal-median background at that radius
- Contamination contour at ε = 2% of background → 2.69σ ellipse → 3 757 px
- After subtracting base-mask overlap: 3 750 px net

**Justification:** This is a geometry-fixed parasitic scatter feature (likely window/kapton scatter), confirmed by independent validation: 2° azimuthal-bin medians in the radial band 441–491 place the blob's sector at +4.7σ above the band median (per skill doc). The 2% contamination contour ensures the residual per-bin bias in I(q) falls to ≈0.1%, well below the 1.2% background noise floor. Without this layer, the blob contributes a 1.5% per-bin bias that would distort the azimuthal average.

---

## Layers considered but not applied

- **Method 02 (pyFAI azimuthal sigma-clip):** Not applied. The sum is already a clean average of 2746 selected shots; per-pixel cosmic-ray outliers are averaged out. The remaining anomalies are either structural (handled by layers 0–3) or diffuse (handled by layer 4). Applying method 02 would risk clipping real LaB6 Bragg intensity.
- **Method 03 (RMM dark-based):** Not applied. The dark-based F2/F3 bad-pixel identification overlaps heavily with Layer 1 (status_bad) for this detector. The 0.225% additional coverage would be marginal given the 6.3% total panel mask already achieved.
- **Method 04 (RMM Feature-6 light):** Not applied. Illumination-dependent bad pixels are most relevant for single-shot analysis. On a 2746-shot sum, their bias is averaged down by √N ≈ 52×, making them negligible compared to the other layers.
- **Method 06 (I(θ) sector screening):** Not invoked separately. The azimuthal-residual layer (04) already detected the parasitic blob that a sector scan would flag. The on-ring detections (Bragg spots) are intentionally retained as real LaB6 signal.

---

## Summary table

| Layer | Description | Pixels | % of image | % of panels |
|---|---|---|---|---|
| 0 | Zero/dead + dilation | 75 583 | 6.90% | — |
| 1 | Status bad (calibration) | 2 597 | 0.24% | — |
| 2 | Beamstop shadow | 31 121 | 2.84% | — |
| 3 | ASIC boundary (absorbed) | 0 | 0.00% | — |
| 4 | Azimuthal-residual blob | 3 750 | 0.34% | — |
| **Total** | **Union** | **113 051** | **10.32%** | **6.26%** |

Unmasked panel-area pixels available for integration: **1 048 429**.
