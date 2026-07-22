# Mask Rationale — Run 475 (Jungfrau1M, LaB6)

**Image:** `sum_assembled.npy` (1064 x 1030), calibrated+normalized sum of kept shots.  
**Output:** `mask_assembled.npy` (1064 x 1030, bool, `True` = masked).

**Final mask:** 164 506 pixels (15.0% of image; 11.2% of real panel area).  
**Unmasked panel pixels:** 931 414.

---

## Layer 1 — Zero/dead pixels + binary dilation

| Metric | Value |
|--------|-------|
| Core (pixels ≤ 0) | 63 306 |
| + 1-iter cross dilation | +56 325 |
| **Layer total** | **119 631** |

**What:** Every pixel with value ≤ 0 in the assembled sum. These are either
geometry-gap cells (exact 0, ~47 000 pixels where no physical detector pixel
exists) or dead/saturated pixels whose calibrated value collapsed to zero or
went negative from pedestal over-subtraction (~16 000 pixels).

**Why:** Zero-gap pixels carry no physical information; negative pixels indicate
calibration failure. A 1-pixel binary dilation (3 x 3 cross structuring element,
1 iteration) extends the mask by one pixel around every core-masked pixel. This
covers the unreliable gap-edge pixels that suffer from charge-sharing with
non-existent neighbors and partial-area interpolation artifacts.

---

## Layer 2 — Known-bad pixels (calib/status\_bad.npy)

| Metric | Value |
|--------|-------|
| Flagged in status\_bad | 4 565 |
| **New beyond Layer 1** | **2 162** |

**What:** Factory pixel-status flags from the Jungfrau calibration store, shape
`(2, 512, 1024)`, mapped into assembled coordinates via `calib/ix.npy` and
`calib/iy.npy`. Any pixel flagged in *any* gain stage is treated as bad.

**Why:** These pixels were identified by the detector vendor or calibration
pipeline as having anomalous response (stuck, noisy, gain-switching errors).
They are the irreducible baseline mask per the
[status-baseline skill](../../skills/masking/00_status_baseline.md). 2 403 of
the 4 565 already overlapped with Layer 1 (dead or negative in the sum); the
remaining 2 162 are pixels that happen to pass the ≤0 cut in this particular
sum but are known to be unreliable.

---

## Layer 3 — Beamstop shadow and beam-center geometry

| Metric | Value |
|--------|-------|
| Shadow band rows | 972–1005 (34 rows) |
| Shadow band cells | 35 020 |
| Extreme-value pixels (|val| > 10⁶) | 547 |
| + 2x 7 x 7 dilation around extremes | included |
| **Layer total** | **60 713** |
| **New beyond Layers 1+2** | **37 987** |

**What:** A horizontal shadow band caused by the beamstop, which sits between
the sample and the detector to block the direct beam. The beam center is near
assembled coordinates **(col 35, row 992)**.

**How the band was identified:**
- Per-row negative-pixel fraction: normal rows have < 1.5% negatives; the
  contiguous band rows 974–1003 jump to 5–31% negatives (up to 315 negative
  pixels per row out of ~1 024 non-gap pixels).
- Per-row median intensity: the shadow rows have median 118–232 ADU vs
  330–410 ADU in neighboring non-shadow rows — a factor of 0.4–0.7×.
- A 2-row margin on each side (rows 972–973 and 1004–1005) covers the
  partially attenuated transition zone where the beamstop penumbra gradually
  attenuates signal.

Additionally, 547 pixels with |value| > 10⁶ (extreme positive or negative
from direct-beam scatter or beamstop-edge diffraction, concentrated near the
beam center at rows 997–1001, cols 28–40) are dilated with a 7 x 7 square
kernel (2 iterations) to cover their local neighborhood.

**Why:** The beamstop attenuates or blocks the diffraction signal in this band.
Pixels in the shadow are unreliable for azimuthal averaging — they read
artificially low and have excess negative values from pedestal subtraction on
near-zero signal. Including them would bias any I(q) profile downward in the
affected q range.

---

## Layer 4 — Double-width ASIC-boundary / panel-edge lines

| Metric | Value |
|--------|-------|
| Edge columns masked | 0, 255, 258, 512, 513, 516, 771, 774, 1029 |
| Edge rows masked | 0, 255, 258, 513, 550, 805, 808, 1063 |
| **Layer total** | **17 744** |
| **New beyond Layers 1–3** | **4 726** |

**What:** Single-pixel lines at detector/module edges and ASIC boundaries. In
the assembled image the inter-module gaps are:

| Gap | Type |
|-----|------|
| Cols 256–257, 514–515, 772–773 | Column gaps (no pixel) |
| Rows 256–257, 514–549, 806–807 | Row gaps (no pixel) |

The pixel immediately adjacent to each gap edge is a "double-width" pixel —
it collects charge from a larger effective area because its neighbor on the
gap side is absent. Measured intensity ratios vs interior pixels (5–10 columns
away):

| Column | Ratio | Column | Ratio |
|--------|-------|--------|-------|
| 0 (edge) | 1.65× | 255 | 2.16× |
| 258 | 2.01× | 513 | 2.01× |
| 516 | 2.16× | 771 | 1.93× |
| 774 | 1.96× | 1029 (edge) | 1.43× |

| Row | Ratio | Row | Ratio |
|-----|-------|-----|-------|
| 0 (edge) | 1.62× | 255 | 2.18× |
| 258 | 2.18× | 513 | 1.77× |
| 550 | 1.45× | 805 | 1.87× |
| 808 | 1.89× | 1063 (edge) | 1.40× |

**Column 512** is a special case: it sits at an ASIC boundary *within* a module
and reads 0.84× of interior intensity (charge split between adjacent ASICs).

**Why:** These elevated (or depressed, for col 512) pixels would bias any
azimuthal average, powder integration, or 2D correlation. They are a known
Jungfrau artifact and the [geometry-gap skill](../../skills/masking/01_geometry_gap.md)
recommends always masking them in assembled-coordinate analyses.

---

## Layers considered but not applied

### Azimuthal sigma-clip (Method 02)
Not applied. This is a signal-dependent mask that removes azimuthal outliers
(Bragg spots from large LaB6 grains, cosmic rays). For this pipeline the
summed image averages over many shots, suppressing shot-level outliers. The
remaining LaB6 Bragg texture is *real signal* that should be preserved for
downstream analysis. If the user intends powder-pattern fitting and wants
outlier-free rings, this layer can be added as a post-hoc step.

### RMM dark-based (Method 03) and RMM Feature-6 (Method 04)
Not applied separately. The RMM masks require the `RobustGaussianFittingLibrary`
and per-ASIC-block robust fitting on calibration constants. The dominant bad
pixels they would catch (0.225% for F2/F3 and 0.156% for F6) substantially
overlap with the pixels already masked by Layer 1 (≤ 0 in sum) and Layer 2
(status_bad). The incremental gain (~700 pixels unique to RMM beyond our
existing layers, based on the skill-file statistics for Run 475) does not
justify the dependency. If the user has pre-built RMM masks available, they
can be unioned trivially.

---

## Summary table

| Layer | Description | New pixels | Cumulative |
|-------|-------------|-----------|------------|
| 1 | Zero/dead + 1-px dilation | 119 631 | 119 631 |
| 2 | status\_bad.npy | +2 162 | 121 793 |
| 3 | Beamstop shadow (rows 972–1005) + beam center | +37 987 | 159 780 |
| 4 | ASIC boundary / edge lines | +4 726 | 164 506 |
| **Total** | | | **164 506 (15.0%)** |

Panel pixels masked: 117 162 / 1 048 576 = **11.2%**.  
Unmasked panel pixels available for analysis: **931 414**.


---

## Layer 5 — Azimuthal-residual blob (skill 05, deterministic application)

| Metric | Value |
|--------|-------|
| Component | (row 686, col 387), radius 466 px, azimuth -42 deg |
| Detection | seed z>5 (peak z=6.2), hysteresis growth to z>3, off-ring policy, r_min=100 |
| **New pixels added** | **905** |

Diffuse parasitic-scattering blob (FWHM ~33 px, +70% over local background) between the
powder rings at radii 367 and 531. Detected per
[skills/masking/05_azimuthal_residual.md](../../skills/masking/05_azimuthal_residual.md).
Independent evidence: its 2-deg azimuthal sector is +4.7 sigma above the radial-band median.
