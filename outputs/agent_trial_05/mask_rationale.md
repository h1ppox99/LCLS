# Mask Rationale — agent_trial_05 (Run 475, Jungfrau1M, LaB6)

**Input:** `sum_assembled.npy` (1064 × 1030, float64) — sum of 2746 kept, calibrated,
normalized shots.

**Output:** `mask_assembled.npy` (1064 × 1030, bool, True = masked).

**Beam center:** (row 992, col 35) in assembled coordinates.

---

## Summary

| Layer | Description | New pixels | Cumul. pixels | Cumul. % (image) |
|-------|-------------|------------|---------------|------------------|
| L0 | Zero/dead + extreme + 1-px dilation | 61 569 | 61 569 | 5.62% |
| L1 | Status-bad from calibration (ix/iy mapped) | 3 846 | 65 415 | 5.97% |
| L2 | Beamstop shadow band (rows 968–1008) + beam center | 41 297 | 106 712 | 9.74% |
| L3 | ASIC boundary double-width lines | 0 | 106 712 | 9.74% |
| L4 | Azimuthal-residual parasitic blob | 2 115 | 108 827 | 9.93% |
| **Total** | | **108 827** | | **9.93% of image** |

**Panel-only coverage:** 61 483 / 1 048 576 = **5.86%** (excluding the 47 369 gap cells
that carry no physical pixel).

---

## Layer-by-layer detail

### Layer 0 — Zero / dead / extreme outliers + 1-px dilation (61 569 px)

**What:** Three sub-populations, then a binary dilation:

1. **Zero pixels (47 369, 4.32%)** — cells in the assembled frame with no physical
   detector pixel mapped to them: inter-module gaps (rows 256–257, 514–549, 806–807)
   and inter-ASIC column gaps (cols 256–257, 514–515, 772–773). These are geometry,
   not signal.

2. **Extreme outlier pixels (563)** — pixels with |value| > 10⁵ in the sum. These are
   dead/hot pixels whose pedestal-subtracted values diverge wildly (e.g., min = −4.6×10⁸
   near beam center). They would dominate any integration bin they land in.

3. **1-pixel binary dilation (+13 637 new)** — a 3×3 structuring element expands the
   combined zero + extreme mask by 1 pixel in all directions. This serves two purposes:

   - **Gap-edge unreliability:** pixels at the physical edge of a module can have partial
     charge collection or cross-talk from the gap region.
   - **ASIC-boundary double-width pixels:** the Jungfrau ASIC layout places physically
     wider pixels at module/ASIC edges. In the assembled sum, these gap-adjacent rows
     and columns show systematically ~2× the normal intensity (e.g., row 255 median =
     646 vs reference ~300; col 255 median = 633 vs ~305). Including them in an
     azimuthal average would create a +100% bias at those radii. The dilation captures
     all of them because every ASIC boundary in this assembled geometry coincides with a
     zero-gap boundary.

**Why this threshold?** |value| > 10⁵ is >120× the p99.5 of the sum (833 ADU). The
553 pixels below −10⁵ are concentrated at the beam center (rows 991–1001, cols 25–45)
where hot pixels accumulate extreme negative pedestals across 2746 shots. The dilation
radius of 1 px is the minimum needed — the double-width effect is confined to a single
pixel adjacent to each gap edge.

---

### Layer 1 — Status-bad from calibration (3 846 new px)

**What:** The detector's factory calibration `pixel_status` array (`calib/status_bad.npy`,
shape (2, 512, 1024), bool) identifies intrinsically defective pixels. These are mapped
from raw panel coordinates to assembled-frame coordinates using the `calib/ix.npy` and
`calib/iy.npy` coordinate lookup tables.

**Pixel accounting:** 4 565 pixels flagged in raw coords → 4 565 mapped to assembled
coords → 3 846 new (719 already covered by Layer 0's zero+dilation mask, mostly pixels
whose extreme values or gap-adjacency already flagged them).

**Why:** This is the always-on baseline per masking method 00 (skills/masking/
00_status_baseline.md). It catches factory-known bad pixels — dead channels, unstable
gain, noisy pre-amps — that may not show extreme values in a sum but still have
unreliable gain calibration. Signal-independent: never removes real diffraction signal.

---

### Layer 2 — Beamstop shadow band + beam center (41 297 new px)

**What:** A horizontal absorber (beamstop arm) casts a shadow across the detector,
attenuating the diffracted signal in a band centered on the beam row. Masked region:

1. **Horizontal band, rows 968–1008 (41 rows × full width):** The row-median intensity
   drops from the reference ~395 ADU (average of rows 940–960 and 1015–1035) to < 355
   in the penumbra (rows 968–970, 1005–1008) and to 83 ADU at the core (row 992, the
   beam row). The 90%-of-reference threshold (< 355 ADU) brackets rows 968–1008.

   Row-by-row validation (selected):
   - row 960: median 411 (normal)
   - row 968: median 344 (−13%, penumbra entry — masked)
   - row 975: median 158 (−60%, deep shadow)
   - row 992: median 83 (−79%, beam center row)
   - row 1005: median 263 (−33%, penumbra exit — masked)
   - row 1009: median 365 (−7%, normal — not masked)

2. **Beam center circle, radius < 25 px:** The direct beam region near (992, 35) where
   the beamstop disc blocks essentially all flux. Most pixels here are already caught by
   the band, but the circle ensures the corners of the beamstop shadow at the detector
   edge are covered.

**Why mask the full penumbra?** Even a 13% intensity deficit biases I(q) in azimuthal
integration. Including partially-shadowed pixels would require a pixel-by-pixel
attenuation correction (which we lack), so masking the entire band is the conservative
and correct choice.

**Why not a narrower band?** The operator hint specified "rows 970–1006." The 90%
threshold extends this by 2 rows on each side (968–1008) to include the measurable
penumbra. This adds ~4×1024 ≈ 4K pixels but removes a systematic 7–13% low bias
from those ring segments.

---

### Layer 3 — ASIC boundary double-width lines (0 new px)

**What:** Jungfrau ASICs are 256×256 pixels. The pixels at ASIC boundaries are
physically wider, collecting charge from a larger area, and appear ~2× brighter than
their neighbors in the assembled sum. These create systematic positive outliers in
azimuthal bins.

**Why 0 new pixels:** In this assembled geometry, every ASIC boundary coincides with a
panel-gap boundary (all-zero rows/columns). Layer 0's 1-pixel dilation of the zero mask
already captures every gap-adjacent double-width pixel. Verified:

- Row 255 (gap-adj): median = 646 → **masked by L0 dilation** ✓
- Row 258 (gap-adj): median = 640 → **masked by L0 dilation** ✓
- Col 255 (gap-adj): median = 633 → **masked by L0 dilation** ✓
- Col 258 (gap-adj): median = 630 → **masked by L0 dilation** ✓
- Rows 805/808, 513/550: similarly masked ✓

This layer exists in the rationale to document that ASIC boundaries were considered and
are covered, not to justify skipping them.

---

### Layer 4 — Azimuthal-residual parasitic blob (2 115 new px)

**What:** A diffuse parasitic-scattering blob at (row 687, col 390), radius ≈ 467 px
from beam center, azimuth ≈ −42°. The blob shows +59% excess intensity over the
azimuthal median at that radius (623.8 ADU vs 391.8 ADU ring median).

**Detection (method 05, skills/masking/05_azimuthal_residual.md):**

1. *Radial-median model:* 1-px radial bins, azimuthal median (robust to the blob
   occupying < 2% of the ring arc), 3-bin boxcar smooth.
2. *Destriping:* subtracted per-column then per-row medians from the residual — this is
   critical (reduces residual noise from ~69 to ~31 ADU per the method documentation).
3. *Gaussian smoothing:* σ = 8 px (matched to FWHM ~33 px target scale).
4. *z-score thresholding:* robust z = (smoothed − median) / (1.4826 × MAD). Seeds at
   z > 5.0 with minimum component area ≥ 150 px; hysteresis growth at z > 3.0.
5. *Off-ring policy:* components with |r_comp − r_ring| > 60 px from any LaB6 ring are
   kept; on-ring components (Bragg spots) are not masked.
6. *Parametric refinement:* 2-D Gaussian fit on the smoothed field with kernel
   deconvolution (σ_true² = σ_fit² − σ_smooth²). The mask extends to the ellipse where
   fitted excess = ε × background (ε = 0.10, i.e., the 10% contamination contour).

**Result:**

- One off-ring component detected: peak z = 5.9, envelope size = 869 px.
- Fitted Gaussian: center (687, 390), σ = (12.8, 13.1) px, amplitude/background = 74.4%.
- Refinement added 1 248 px beyond the envelope, extending the mask to the 10%
  contamination contour — total layer = 2 115 px (0.20% of panel area).
- r_min = 100 px excluded a spurious near-beam-center residual from the blob layer.

**Why mask it?** The blob is parasitic scatter (likely from a window, kapton, or beamline
optic), not diffraction signal. Without masking, it biases I(q) at q ≈ 2π/d
corresponding to r ≈ 467 px. Method 05's validation (on Run0475 trial_02) showed
the envelope-only mask leaves a 1.5% per-bin bias above the 1.2% noise floor; the ε=0.10
contour reduces this to 0.52% (below floor/2), which is the quality target.

---

## Layers considered but not applied

| Method | Reason skipped |
|--------|----------------|
| 02 (pyFAI azimuthal sigma-clip) | Signal-dependent per-pixel method that removes Bragg peaks and texture arcs. Not appropriate for a LaB6 calibrant where Bragg rings ARE the signal of interest. The defect-level outliers are already caught by L0 + L1. |
| 03 (RMM dark-based) | Requires the raw dark-calibration constants (pedestal, rms per ASIC block). The necessary `ped.npy` and `gain.npy` exist in calib/ but are gain-corrected constants, not the per-ASIC-block statistics needed for the MSSE robust fit. L0 (extreme outliers) and L1 (status-bad) already cover the factory-known defects that RMM-dark targets. Adding this layer would require re-deriving the per-block statistics, for an expected yield of ~700 additional pixels (per the method file: 2363 total, 1661 overlap with existing masks). |
| 04 (RMM Feature-6 light) | Catches illumination-dependent bad pixels. Requires comparison of lit vs dark frames at the per-ASIC-block level. Similar dependency as method 03. Expected yield ~0.156% ≈ 1600 px, mostly overlapping L0/L1. |
| 06 (I(θ) sector screening) | Screening tool that routes detections to other layers. The one actionable feature it would detect (the parasitic blob) is already handled by L4. |

---

## Quality check

- **mask_log.json** confirms: 108 827 masked pixels, 5.86% of panel area, 987 093 unmasked
  panel pixels remaining.
- **masked_sum.png** shows clean powder rings with all known artifacts (gaps, beamstop
  shadow, hot/dead pixels, parasitic blob) in red.
- The LaB6 ring positions at r ≈ 367, 531, 734, 847 px are unaffected by the mask — no
  ring-coincident features were masked.
- The beamstop shadow band is fully excised (no partial-attenuation artifacts remain).
