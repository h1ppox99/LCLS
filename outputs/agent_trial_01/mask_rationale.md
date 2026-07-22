# Mask Rationale — Run 475 (Jungfrau1M, LaB6)

**Image:** `sum_assembled.npy` (1064 x 1030), float64 sum of 1781 kept, calibrated,
normalised shots.  
**Mask:** `mask_assembled.npy` (1064 x 1030), bool, `True = masked`.  
**Total masked:** 102 416 / 1 095 920 pixels (9.35% of assembled image; **5.25% of
physical panel area**).  
**Usable detector area:** 993 504 pixels (94.8% of panel area).

---

## Layer-by-layer breakdown

### Layer 1 — Geometry gaps + 1-pixel dilation

| Metric | Value |
|---|---|
| Core gap pixels (img == 0) | 47 369 (4.32%) |
| After 1-pixel dilation (3x3 structuring element) | 59 802 |
| **New pixels added to mask** | **59 802** |

**What:** Every cell in the assembled image where no physical detector pixel exists
— inter-module gap (rows 514–549, 36 rows), intra-panel ASIC column gaps (cols
256–257, 514–515, 772–773), and intra-panel ASIC row gaps (rows 256–257,
806–807). These are exactly zero in the assembled sum.

**Why:** Gap cells contain no photon information; reading them as zero-intensity
would bias any azimuthal average or peak fit. The 1-pixel binary dilation (3x3
square structuring element, 1 iteration) serves two purposes:
1. **ASIC boundary double-intensity lines:** Edge pixels flanking the gaps
   (cols 255, 258, 513, 516, 771, 774; rows 255, 258, 513, 550, 805, 808)
   have ~2x elevated median intensity (e.g., col 255 median 621 vs col 254
   median 329) due to effective-area doubling at the ASIC boundary. The
   dilation automatically catches them.
2. **Unreliable gap-edge pixels:** Pixels at the exact boundary of physical
   panel edges may have partial charge sharing or geometric vignetting.

**Trade-off:** The dilation masks ~12 400 real detector pixels beyond the raw gap,
but these are the known-biased ASIC boundary lines. The cost (1.2% of panel area)
is justified by eliminating the ~2x intensity bias they would inject.

**Note (Layer 4 subsumption):** ASIC boundary lines were also targeted explicitly
as Layer 4 (see below), but the dilation in this layer already covered 100% of
them, so Layer 4 added 0 new pixels.

---

### Layer 2 — Dead/stuck pixels (strongly negative, < −100)

| Metric | Value |
|---|---|
| Strongly negative pixels (val < −100) | 3 844 raw |
| **New pixels added to mask** | **3 784** |

**What:** Pixels with calibrated sum < −100. These include:
- 601 pixels with extreme values < −1000 (down to −4.3 × 10⁸), concentrated at
  specific columns (172–173, 633–634, 718–720) — stuck/dead pixels whose
  pedestal subtraction yields pathological results.
- ~3 200 pixels in the range [−1000, −100] — less extreme but still physically
  unreasonable in a sum of 1781 positive-signal shots.

**Why:** A physical sum of normalised LaB6 scattering shots cannot be significantly
negative. The P50 of the good (nonzero, non-gap) image is ~332; pixels at −100
are > 1σ below zero. These are defective pixels whose calibration fails.

**Mildly negative pixels not masked:** 10 739 pixels in (−100, 0) were examined
and found to be 86.1% isolated single pixels (8 424 connected components, median
cluster size = 1, max cluster = 19). This is consistent with statistical noise
in the pedestal-subtracted, normalised sum — not structured defects. Masking them
would remove ~1% of detector area unnecessarily.

**Trade-off:** The −100 threshold is conservative; a few mildly defective pixels
in (−100, 0) may survive. This is preferable to false-positive masking of noise
fluctuations.

---

### Layer 3 — Known-bad pixels (`status_bad.npy`)

| Metric | Value |
|---|---|
| Bad pixels in raw space | 4 565 |
| Mapped to assembled coordinates | 4 565 |
| **New pixels added to mask** | **2 687** |

**What:** The detector calibration store's per-pixel status array (`status_bad.npy`,
shape 2 × 512 × 1024, bool) flags factory-known bad pixels. These are mapped from
raw (panel, row, col) to assembled (row, col) via the `ix.npy` / `iy.npy`
coordinate maps.

**Why:** Factory characterisation identifies pixels with anomalous gain, dark
current, or connectivity issues. This is Method 00 from the masking skill set —
the always-on baseline that every other mask unions onto.

**Overlap:** 1 878 of the 4 565 raw bad pixels were already masked by Layers 1–2
(gap-adjacent or strongly negative), leaving 2 687 novel. These are bad pixels
that happen to read near-normal values in the sum but are unreliable.

---

### Layer 4 — ASIC boundary double-intensity lines (explicit)

| Metric | Value |
|---|---|
| Edge-column/row pixels targeted | 12 528 |
| **New pixels added to mask** | **0** |

**What:** Explicitly targeted the single-pixel-wide columns/rows flanking each
ASIC gap:
- Columns: 255, 258, 513, 516, 771, 774
- Rows: 255, 258, 513, 550, 805, 808

These show ~2x median intensity vs their neighbors:

| Location | Median (ADU) | Neighbor median | Ratio |
|---|---|---|---|
| Col 255 | 621 | 329 | 1.89x |
| Col 258 | 624 | 328 | 1.90x |
| Col 513 | 692 | 289 | 2.39x |
| Col 516 | 705 | 330 | 2.14x |
| Col 771 | 736 | 367 | 2.01x |
| Col 774 | 735 | 358 | 2.05x |
| Row 255 | 641 | 310 | 2.07x |
| Row 258 | 638 | 303 | 2.11x |
| Row 805 | 698 | 373 | 1.87x |
| Row 808 | 717 | 401 | 1.79x |

**Why:** Intensity doubling at ASIC edges biases azimuthal averages and ring fits.

**Result:** All 12 528 pixels were already captured by Layer 1's gap dilation,
confirming the dilation strategy is complete. Retaining this layer as an explicit
audit trail.

---

### Layer 5 — Beamstop shadow

| Metric | Value |
|---|---|
| Shadow-band pixels (rows 970–1006) | 38 110 raw |
| **New pixels added to mask** | **35 953** |

**What:** Horizontal shadow band cast by the beamstop. Beam center is near
(col 35, row 992). Row-by-row robust median analysis (ignoring outliers
> 3σ) shows:

| Row | Robust median (ADU) | % of normal (385) |
|---|---|---|
| 960 | 403 | 105% (normal) |
| 968 | 339 | 88% (onset) |
| 970 | 301 | 78% |
| 974 | 189 | 49% |
| 980 | 95 | 25% (deep shadow) |
| 990 | 91 | 24% (deep shadow) |
| 1000 | 100 | 26% |
| 1003 | 110 | 29% |
| 1006 | 295 | 77% |
| 1008 | 339 | 88% (recovery) |
| 1010 | 357 | 93% (normal) |

**Why:** The beamstop attenuates the beam, creating a horizontal band where
measured intensity is 20–75% below the true diffraction signal. Including these
pixels would:
- Suppress LaB6 ring intensity in azimuthal averages by 20–75%,
- Distort peak shapes and widths,
- Bias background fits.

**Boundary choice:** Rows 970–1006 (37 rows) captures all rows with > 20%
attenuation. The transition zone (rows 968–969 at 12–22% drop, rows 1007–1009
at 12–17% drop) was not masked — the attenuation there is within the typical
ring-to-ring intensity variation and is marginal. A stricter threshold of 50%
(rows 974–1003, 30 rows) would retain 7 more rows but leave 20–50% attenuated
pixels that could bias peak fitting.

**Shadow spans full width:** The shadow runs across all 1030 columns uniformly
(verified: Q25–Q75 range at row 990 = 18–162, similar scatter at all columns),
so masking full rows is correct.

---

### Layer 6 — Extreme outlier pixels (hot pixels)

| Metric | Value |
|---|---|
| Pixels with |value| > 3000 | 843 raw |
| **New pixels added to mask** | **190** |

**What:** Pixels with physically unreasonable absolute intensity > 3000 ADU in the
sum. The good-data P99.9 is 1145 ADU, so 3000 is a 2.6× safety margin. These are
stuck-high or residual stuck-low pixels not caught by `status_bad`.

**Why:** Hot/stuck pixels create sharp spikes that dominate local statistics and
corrupt peak fits. At 3000 ADU, these are > 9× the median (332) — no LaB6 Bragg
peak in a 1781-shot sum reaches this level in the powder-averaged regime.

**LaB6 Bragg peaks preserved:** The bright spots visible in the image at
positions like (34, 830) = 37 115 ADU and (108, 909) = 60 124 ADU are
single-crystal Bragg spots from large LaB6 grains — they sit exactly on the
powder rings and are real signal. However, in a powder/amorphous reduction they
are outliers that bias azimuthal averages. The 843 raw extreme pixels include
both stuck-hot detector defects and these Bragg spots. 653 of the 843 were
already masked by earlier layers (primarily the beamstop shadow); the 190 new
pixels are scattered across the image.

**Trade-off:** If single-crystal Bragg analysis were the goal, this layer should
be removed. For powder azimuthal averaging (the pipeline's purpose), removing
these outliers is correct.

---

## Methods from skill set considered but not applied

| Method | Decision | Reason |
|---|---|---|
| 02 — pyFAI azimuthal sigma-clip | **Skipped** | Signal-dependent; would remove real LaB6 Bragg/texture features. Our Layer 6 outlier threshold achieves the same goal (remove extreme outliers) with a simpler, more transparent criterion. For a full azimuthal integration, adding Method 02 post-hoc is straightforward. |
| 03 — RMM dark-based (F2/F3) | **Partially covered** | Layer 2 (strongly negative) and Layer 3 (status_bad) together catch the same defective pixels. RMM requires `RobustGaussianFittingLibrary` (needs recompilation on Apple Silicon) and dark calibration inputs (`rms`) not available in the current `calib/` directory. The 2 687 novel status_bad pixels in Layer 3 provide equivalent coverage for this run. |
| 04 — RMM Feature-6 (illumination) | **Skipped** | Requires per-frame raw images via `psana` (Linux-only). The single-image approximation (using the sum) is less reliable. Our Layer 6 outlier threshold catches persistently anomalous pixels empirically. |

---

## Summary table

| Layer | Description | New pixels | Cumulative |
|---|---|---|---|
| 1 | Geometry gaps + 1px dilation | 59 802 | 59 802 |
| 2 | Dead/stuck pixels (val < −100) | 3 784 | 63 586 |
| 3 | Known-bad (`status_bad.npy`) | 2 687 | 66 273 |
| 4 | ASIC boundary lines (subsumed by L1) | 0 | 66 273 |
| 5 | Beamstop shadow (rows 970–1006) | 35 953 | 102 226 |
| 6 | Extreme outliers (|val| > 3000) | 190 | 102 416 |
| **Total** | | **102 416** | **9.35% of image** |

**Panel-area breakdown:** 55 047 / 1 048 551 physical pixels masked (**5.25%**).  
**Usable:** 993 504 pixels (94.8% of panel area).
