---
name: xray-masking-azimuthal-sector-itheta
description: Shape-agnostic azimuthal screening — build I(theta) curves per radial band on the destriped residual, robust-z the (band, sector) cells with BOTH signs, hysteresis-cluster the outliers. Catches any azimuthal-symmetry violation at fixed r — parasitic blobs, arcs, streaks, shadows, AND negative deficits/ASIC-block offsets that the positive-only blob detector (method 05) ignores. Detection front-end; hand blob-like positives to method 05's contamination contour for the final footprint.
---

# Masking · Method 6 — azimuthal I(θ) sector screening (signal-based)

**Family:** signal-dependent. **Part of:** [masking](README.md).
**Relationship to [05](05_azimuthal_residual.md):** 06 is the *screener* (1-D, shape-agnostic,
two-signed), 05 is the *footprint builder* (2-D matched filter + parametric contour, positive
excesses). Run 06 to find and localize every violation of powder symmetry; mask blob-like
positives via 05's Gaussian-contour refinement; report the rest (see classification below).

## Principle

At fixed radius, powder scattering is flat in azimuth: any structure in a band's I(θ) curve
is an anomaly. Three preprocessing steps make the curve judgeable:

```
resid    : skill-05 residual (radial-median model removed, per-col/per-row destriped)
           — plain I(θ) on the raw sum FAILS: column-gain stripes appear as steps in θ
cells    : (band Δr=15 px) × (sector Δθ=2°) medians of resid, ≥60 px per cell
detrend  : per band, subtract a rolling median over θ (window 15 bins = 30°)
           — absorbs real smooth asymmetries (vignetting, broad diffuse anisotropy);
             compact anomalies (a blob spans ~5 bins) survive
z-score  : robust global (median/MAD over all cells); use |z| — keep BOTH signs
detect   : seeds |z| > 4.5  →  binary_propagation into |z| > 2.5  →  clusters ≥ 2 cells
policy   : skip ring bands (|r_band − r_ring| ≤ 25) — Bragg texture is real signal
```

## Classification of detections (what to do with each)

| Signature | Physical meaning | Action |
|---|---|---|
| **positive, compact** (few cells, one band-group) | parasitic scattering blob / ghost | mask — hand to method 05's ε-contour for a shape-faithful footprint |
| **positive, elongated along θ** | scatter arc / streak (e.g. reflection line) | mask the flagged cells + 1-cell margin |
| **negative, aligned with ASIC/panel boundaries** | residual block gain/pedestal offset the row/col destripe cannot remove | do NOT silently mask — report in `mask_rationale.md`; route to calibration/normalization discussion; mask only if the deficit biases target bins beyond the noise floor |
| **negative, full-width band** | missed beamstop/shadow structure | route to the beamstop layer |

## Parameters (tested on Run0475, agent_trial_03 sum)

| Param | Value | Why |
|---|---|---|
| band Δr / sector Δθ | 15 px / 2° (screen) → 5–8 px (fine pass) | see "Radial resolution" below — Δr is a screening-robustness vs footprint-sharpness trade-off |
| `min_px` | 60 | median of fewer pixels is too noisy at 2° sectors |
| destripe first | **required** | without it: column-gain steps in θ → false steps (σ inflated to 33 ADU, mass false positives). With destripe only: broad real asymmetry survives → a 53-cell false cluster. Destripe **+ θ-detrend**: σ = 20 ADU, clean |
| θ-detrend window | 15 bins (30°) | wide enough to pass a ≤10-bin anomaly, narrow enough to track real smooth structure |
| seed / grow / min cluster | 4.5σ / 2.5σ / 2 cells | on Run0475: 9 seeds → 3 clusters, zero singles kept |
| ring-band margin | 25 px | band-level analogue of 05's `ring_margin` |
| signs | both (use \|z\|) | negative anomalies (shadows, block deficits) are equally symmetry violations — method 05 cannot see them |

## Radial resolution (Δr): two-scale guidance

Δr sweep on Run0475 (strong blob + two weak |z|≈5 negative block offsets):

| Δr (min_px) | blob | weak clusters surviving | blob-cell footprint vs 05 ellipse |
|---|---|---|---|
| 15 (60) | z=11.2, 7 cells | A−, B− | 1 714 px, 47 % |
| 10 (60) | z=11.6, 11 cells | B−, weak + at r≈767 | 1 785 px, 49 % |
| 8 (50) | z=11.2, 12 cells | A− only | 1 566 px, 43 % |
| 5 (40) | z=10.3, 20 cells | none | 1 626 px, 45 % |

Three lessons:

1. **Partition robustness separates strong-real from borderline.** The blob holds z ≈ 11 at
   every grid; the |z|≈5 block offsets flicker in and out as the cell boundaries move.
   Confirm a weak cluster only if it survives ≥ 2 different grids.
2. **Finer Δr sharpens radial edges but does NOT extend coverage.** Blob-vs-ellipse coverage
   stays ~45 % at every Δr: the diffuse wings sit below the `z_grow` support in any cell
   size (σ_cell grows as cells shrink). Footprints for blob-like positives always come from
   05's contamination contour; 06's fine pass only localizes.
3. **Inner-radius validity shrinks with Δr** — a cell needs arc(r)·Δr ≥ `min_px`, so
   Δr=5/min_px=40 only covers r ≳ 230 px, vs r ≳ 120 px at Δr=15.

Recommended flow: **screen at Δr=15** (full range, stable statistics) → **re-scan each
detection at Δr=5–8** for radial localization and the robustness check of lesson 1.

## Reference implementation (numpy/scipy only)

```python
import numpy as np
from scipy import ndimage

def itheta_sector_scan(resid, base_mask, bc=(992.0, 35.0), r0=120, r1=1100,
                       th0=-100.0, th1=10.0, dr=15, dth=2.0, min_px=60,
                       z_seed=4.5, z_grow=2.5, min_cells=2,
                       ring_radii=(367, 531, 734, 847, 1245), ring_margin=25,
                       detrend_win=15):
    """resid = skill-05 destriped residual (NaN where base_mask). Returns
    (clusters, zmap, kept_cells, pixel_layer)."""
    H, W = resid.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - bc[0], cc - bc[1])
    th = np.degrees(np.arctan2(rr - bc[0], cc - bc[1]))
    nb, ns = int((r1 - r0) / dr), int((th1 - th0) / dth)
    bi = ((rad - r0) / dr).astype(int); si = ((th - th0) / dth).astype(int)
    inside = (rad >= r0) & (rad < r1) & (th >= th0) & (th < th1) & np.isfinite(resid)
    cid = bi * ns + si
    order = np.argsort(cid[inside], kind="stable")
    fs, vs = cid[inside][order], resid[inside][order]
    edges = np.searchsorted(fs, np.arange(nb * ns + 1))
    med = np.full((nb, ns), np.nan)
    for c in range(nb * ns):
        a, b = edges[c], edges[c + 1]
        if b - a >= min_px: med[c // ns, c % ns] = np.median(vs[a:b])
    det = np.full_like(med, np.nan)                      # theta detrend per band
    h = detrend_win // 2
    for i in range(nb):
        for j in range(ns):
            seg = med[i, max(0, j - h):j + h + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 5 and np.isfinite(med[i, j]):
                det[i, j] = med[i, j] - np.median(seg)
    gm = np.nanmedian(det)
    sig = 1.4826 * np.nanmedian(np.abs(det - gm))
    z = (det - gm) / sig
    ring_band = np.array([any(abs(r0 + (i + .5) * dr - rp) <= ring_margin
                              for rp in ring_radii) for i in range(nb)])
    zf = np.nan_to_num(np.abs(z))
    seeds = (zf > z_seed) & ~ring_band[:, None]
    grown = ndimage.binary_propagation(seeds, mask=(zf > z_grow) & ~ring_band[:, None])
    lab, ncl = ndimage.label(grown)
    kept = np.zeros_like(grown); clusters = []
    for k in range(1, ncl + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < min_cells: continue
        kept[ys, xs] = True
        sign = float(np.sign(np.nanmedian(det[ys, xs])))
        clusters.append({"cells": int(len(ys)),
                         "r": float(r0 + (ys.mean() + .5) * dr),
                         "theta": float(th0 + (xs.mean() + .5) * dth),
                         "max_abs_z": float(zf[ys, xs].max()), "sign": sign})
    sel = np.zeros((H, W), bool)
    okp = inside.copy()
    sel[okp] = kept[np.clip(bi, 0, nb - 1)[okp], np.clip(si, 0, ns - 1)[okp]]
    layer = sel & ~base_mask
    return clusters, z, kept, layer
```

## Result (Run0475, agent_trial_03 sum, blob un-masked for the test)

Three clusters, all physically interpretable, **zero noise false positives**:

| cluster | r, θ | max\|z\| | sign | classification |
|---|---|---|---|---|
| blob | 468 px, −41° | **11.2** (strongest) | + | parasitic blob → mask via 05 (ε-contour 3 627 px) |
| A | 284 px, −44° | 5.0 | − | ASIC-block offset (~−35 ADU, boundary at rows 806–807) → report |
| B | 1001 px, −25° | 5.4 | − | right-edge column-block offset (~−55 ADU) → report |

A and B are invisible to method 05 by design (positive seeds only) — 06 is the only method
in this set that surfaces them.

## Report contract

After running the scan, write **`outputs/<run>/_itheta_screen.json`**:

```json
{
  "skill": "skills/masking/06_azimuthal_sector_itheta.md",
  "grid": {"dr_px": 15, "dtheta_deg": 2.0, "z_seed": 4.5, "z_grow": 2.5, "min_cells": 2},
  "clusters": [
    {"r_px": 468, "theta_deg": -41, "cells": 7, "max_abs_z": 11.2, "sign": 1,
     "classification": "parasitic blob", "action": "masked via method-05 contour"},
    {"r_px": 284, "theta_deg": -44, "cells": 5, "max_abs_z": 5.0, "sign": -1,
     "classification": "ASIC-block offset", "action": "reported (calibration systematics)"}
  ]
}
```

Every cluster carries a `classification` and an `action` per the table above, and the same
findings are summarized in `mask_rationale.md` — the report is what makes "nothing found"
distinguishable from "nobody looked", and it is the channel through which negative
(calibration) findings reach the humans.

## When to use

Routine screening of any summed assembled image before azimuthal integration; verifying that
a mask left no symmetry violations (the max off-ring |z| of this map is a natural scalar for
a **verifier criterion** — "azimuthal uniformity"); hunting non-Gaussian contamination
(arcs, streaks, shadows) that a blob-matched filter misses.

## Trade-offs

Cell granularity (15 px × 2°) makes the raw footprint blocky and under-covers diffuse wings —
for blob-like positives always refine with 05's contamination contour instead of masking the
cells directly. θ coverage is partial and r-dependent (off-center beam). Negative detections
are calibration diagnostics, not automatic mask layers — masking them without thought hides a
flat-field problem the normalization side should know about.

## Repo

Prototype + validation artifacts: `outputs/agent_trial_03_refined/_itheta_*.npy`,
`_itheta_method.png` (I(θ) curve, |z| map, footprint comparison),
`_itheta_candidates.png` (visual inspection of the two negative block offsets).
