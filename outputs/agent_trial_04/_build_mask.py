#!/usr/bin/env python3
"""Build the multi-layer detector mask for agent_trial_04 (Jungfrau1M, LaB6 run 475).

Layers:
  0  zero/dead + gap-edge dilation
  1  known-bad pixels from calib/status_bad.npy
  2  beamstop shadow band
  3  ASIC boundary double-width columns (explicit, for accounting)
  4  azimuthal-residual blob detection (method 05)

Writes:  mask_assembled.npy  (1064, 1030) bool, True = masked
         _mask_layers.json   per-layer pixel counts
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
import json, sys, os

OUT = "outputs/agent_trial_04"
img = np.load(f"{OUT}/sum_assembled.npy")
H, W = img.shape  # (1064, 1030)
assert (H, W) == (1064, 1030), f"Unexpected shape {img.shape}"

BC_ROW, BC_COL = 992.0, 35.0   # beam center, assembled coords
rr, cc = np.mgrid[0:H, 0:W]
rad = np.hypot(rr - BC_ROW, cc - BC_COL)

layer_stats = {}

# =========================================================================
# LAYER 0 — Zero / dead pixels + gap-edge dilation
# =========================================================================
gap_mask  = (img == 0)                     # panel gaps, geometry holes
dead_mask = (img < 0)                      # negative: dead / gain-switch overflow

# Dilate gaps by 1 px to cover unreliable gap-edge pixels.
# This also catches the double-bright ASIC boundary columns/rows
# flanking each gap (cols 255,258,513,516,771,774; rows 255,258,513,550,805,808).
gap_dilated = ndimage.binary_dilation(gap_mask, iterations=1)
layer0 = gap_dilated | dead_mask
ct0 = int(layer0.sum())
layer_stats["0_zero_dead_dilation"] = {
    "pixels": ct0,
    "pct_of_image": round(ct0 / (H * W) * 100, 3),
    "detail": {
        "gap_pixels_exact_zero": int(gap_mask.sum()),
        "negative_pixels": int(dead_mask.sum()),
        "gap_dilated": int(gap_dilated.sum()),
    }
}
print(f"Layer 0 (zero/dead + dilation): {ct0} px ({ct0/(H*W)*100:.2f}%)")

# =========================================================================
# LAYER 1 — Known-bad pixels from calib/status_bad.npy
# =========================================================================
status = np.load("calib/status_bad.npy")   # (2, 512, 1024) bool
ix = np.load("calib/ix.npy")              # (2, 512, 1024) uint16 → assembled row
iy = np.load("calib/iy.npy")              # (2, 512, 1024) uint16 → assembled col

status_assembled = np.zeros((H, W), dtype=bool)
for panel in range(2):
    bad_px = status[panel]                 # (512, 1024) bool
    r_idx = ix[panel][bad_px]
    c_idx = iy[panel][bad_px]
    status_assembled[r_idx, c_idx] = True

layer1 = status_assembled & ~layer0       # new pixels only
ct1 = int(layer1.sum())
layer_stats["1_status_bad"] = {
    "pixels": ct1,
    "pct_of_image": round(ct1 / (H * W) * 100, 3),
    "total_raw_bad": int(status.sum()),
    "mapped_to_assembled": int(status_assembled.sum()),
}
print(f"Layer 1 (status_bad): {ct1} px ({ct1/(H*W)*100:.3f}%)")

# =========================================================================
# LAYER 2 — Beamstop shadow band
# =========================================================================
# Row medians drop to 27-52% of global (305.5) in rows 972-1005.
# Shadow onset is gradual: row 970 (101%) → row 972 (86%) → row 976 (46%)
# Recovery: row 1004 (70%) → row 1006 (101%).
# Mask rows 970-1006 inclusive (conservative edges include the transition zone
# where partial shadow biases the azimuthal average).
beamstop = np.zeros((H, W), dtype=bool)
beamstop[970:1007, :] = True               # rows 970 through 1006

layer2 = beamstop & ~(layer0 | layer1)
ct2 = int(layer2.sum())
layer_stats["2_beamstop_shadow"] = {
    "pixels": ct2,
    "pct_of_image": round(ct2 / (H * W) * 100, 3),
    "row_range": "970-1006",
}
print(f"Layer 2 (beamstop shadow): {ct2} px ({ct2/(H*W)*100:.2f}%)")

# =========================================================================
# LAYER 3 — ASIC boundary double-width columns (explicit)
# =========================================================================
# Empirically verified: cols 255, 258, 513, 516, 771, 774 show ~2× median
# intensity (632–751 ADU vs ~320 neighbors). These are the data columns
# flanking each gap column pair (256-257, 514-515, 772-773).
# Layer 0 dilation already covers these because they are gap-adjacent.
# This layer is kept for accounting clarity; new-pixel count should be ≈0.
asic = np.zeros((H, W), dtype=bool)
for c in [255, 258, 513, 516, 771, 774]:
    asic[:, c] = True

prior = layer0 | layer1 | layer2
layer3 = asic & ~prior
ct3 = int(layer3.sum())
layer_stats["3_asic_boundary"] = {
    "pixels": ct3,
    "pct_of_image": round(ct3 / (H * W) * 100, 3),
    "note": "double-bright ASIC-boundary columns; mostly absorbed by layer-0 dilation",
    "cols": [255, 258, 513, 516, 771, 774],
}
print(f"Layer 3 (ASIC boundary): {ct3} px ({ct3/(H*W)*100:.3f}%)")

# =========================================================================
# LAYER 4 — Azimuthal-residual blob detection (method 05)
# =========================================================================
# Build the azimuthal-median radial model, subtract, destripe, smooth, threshold.
# Parameters from skill doc (validated on Run0475).
base_mask = layer0 | layer1 | layer2 | layer3

# Ring radii for LaB6 (from skill doc)
RING_RADII = (367, 531, 734, 847, 1245)
SIGMA_SMOOTH = 8.0
Z_SEED = 5.0
Z_GROW = 3.0
MIN_SIZE = 150
RING_MARGIN = 60
R_MIN = 100.0
REFINE_EPS = 0.02

# Build safe image (mask out base_mask and extreme outliers)
safe = np.where(base_mask | (np.abs(img) > 1e5), np.nan, img)

# Radial binning
rbin = rad.astype(int)
ok = ~np.isnan(safe)

# Azimuthal median model per radial bin
max_r = rbin.max() + 1
med_r = np.full(max_r, np.nan)
for i in range(max_r):
    sel = ok & (rbin == i)
    cnt = sel.sum()
    if cnt >= 40:
        med_r[i] = np.nanmedian(safe[sel])

# 3-bin boxcar smoothing of the radial model
valid_r = ~np.isnan(med_r)
k = np.ones(3) / 3.0
med_s = med_r.copy()
conv_val = np.convolve(np.nan_to_num(med_r), k, "same")
conv_wt  = np.convolve(valid_r.astype(float), k, "same")
med_s[valid_r] = conv_val[valid_r] / np.maximum(conv_wt[valid_r], 1e-9)

# Residual
resid = safe - med_s[rbin]

# Destripe: column then row medians (CRITICAL per skill doc)
resid -= np.nanmedian(resid, axis=0, keepdims=True)
resid -= np.nanmedian(resid, axis=1, keepdims=True)

ok2 = ~np.isnan(resid)

# Gaussian smoothing (NaN-aware via coverage weighting)
sm = ndimage.gaussian_filter(np.nan_to_num(resid), SIGMA_SMOOTH)
w  = ndimage.gaussian_filter(ok2.astype(float), SIGMA_SMOOTH)
sm = np.where(w > 0.6, sm / np.maximum(w, 1e-9), np.nan)

# Robust z-score
sm_med = np.nanmedian(sm)
sm_mad = np.nanmedian(np.abs(sm - sm_med))
zmap = (sm - sm_med) / (1.4826 * sm_mad + 1e-12)
zf = np.nan_to_num(zmap)

# Seed detection: connected components above z_seed with area >= min_size
lab, n_lab = ndimage.label(zf > Z_SEED)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n_lab + 1))
big_enough = [i + 1 for i, s in enumerate(sizes) if s >= MIN_SIZE]
seeds = np.isin(lab, big_enough)

# Hysteresis growth: propagate seeds into z > z_grow region
grown = ndimage.binary_propagation(seeds, mask=(zf > Z_GROW))

# Component filtering: off-ring policy, r_min exclusion
glab, gn = ndimage.label(grown)
layer4 = np.zeros((H, W), dtype=bool)
kept_components = []

print(f"\n  Blob detection: {gn} grown components found")
for g in range(1, gn + 1):
    comp = (glab == g)
    comp_area = int(comp.sum())
    r_comp = float(rad[comp].mean())
    z_peak = float(zf[comp].max())

    if r_comp < R_MIN:
        print(f"    comp {g}: r={r_comp:.0f} < r_min={R_MIN}, SKIP (beam-center territory)")
        continue

    on_ring = any(abs(r_comp - rp) <= RING_MARGIN for rp in RING_RADII)
    if on_ring:
        print(f"    comp {g}: r={r_comp:.0f}, area={comp_area}, z_peak={z_peak:.1f} — ON-RING, skip")
        continue

    print(f"    comp {g}: r={r_comp:.0f}, area={comp_area}, z_peak={z_peak:.1f} — OFF-RING, KEEP")
    kept_components.append(comp)
    layer4 |= comp

# Parametric refinement: fit 2-D Gaussian, extend mask to contamination contour
refined_info = []
for ci, comp in enumerate(kept_components):
    ys, xs = np.where(comp)
    y0g, x0g = float(ys.mean()), float(xs.mean())
    r_eq = np.sqrt(comp.sum() / np.pi)
    half = int(np.clip(2.5 * r_eq + 2 * SIGMA_SMOOTH, 60, 300))
    sl = (slice(max(0, int(y0g) - half), min(H, int(y0g) + half)),
          slice(max(0, int(x0g) - half), min(W, int(x0g) + half)))

    Zc = sm[sl]
    Yc = rr[sl]
    Xc = cc[sl]
    good = np.isfinite(Zc)

    if good.sum() < 500:
        refined_info.append({"component": ci, "status": "skip_too_few_pixels"})
        continue

    def gmod(p, x, y):
        A, x0, y0, sa, sb, off = p
        return A * np.exp(-(x - x0)**2 / (2 * sa**2)
                          - (y - y0)**2 / (2 * sb**2)) + off

    s0 = float(np.clip(r_eq, 1.2 * SIGMA_SMOOTH, 100.0))
    try:
        res = least_squares(
            lambda p: gmod(p, Xc[good], Yc[good]) - Zc[good],
            [float(np.nanmax(Zc)), x0g, y0g, s0, s0, 0.0],
            loss="soft_l1", f_scale=30)
        p = res.x
    except Exception as e:
        refined_info.append({"component": ci, "status": f"fit_failed: {e}"})
        continue

    A_f, x0, y0, sa_f, sb_f, _ = p
    # Deconvolve smoothing kernel
    sa_t = np.sqrt(max(sa_f**2 - SIGMA_SMOOTH**2, 9.0))
    sb_t = np.sqrt(max(sb_f**2 - SIGMA_SMOOTH**2, 9.0))
    A_t = A_f * (abs(sa_f) * abs(sb_f)) / (sa_t * sb_t)

    r_center = min(int(np.hypot(y0 - BC_ROW, x0 - BC_COL)), len(med_s) - 1)
    bg = med_s[r_center]

    if not (np.isfinite(bg) and bg > 0 and A_t > REFINE_EPS * bg):
        refined_info.append({
            "component": ci, "status": "below_eps_threshold",
            "A_t": float(A_t), "bg": float(bg) if np.isfinite(bg) else None
        })
        continue

    g_r = np.sqrt(2 * np.log(A_t / (REFINE_EPS * bg)))
    ellipse = (((cc - x0) / (sa_t * g_r))**2
               + ((rr - y0) / (sb_t * g_r))**2) <= 1
    layer4 |= ellipse
    refined_info.append({
        "component": ci, "status": "refined",
        "center": (float(y0), float(x0)),
        "sigma_true": (float(sa_t), float(sb_t)),
        "amplitude_frac": float(A_t / bg),
        "contour_radius_px": float(g_r),
        "ellipse_area_px": int(ellipse.sum()),
    })
    print(f"    Refined comp {ci}: center=({y0:.0f},{x0:.0f}), "
          f"σ=({sa_t:.1f},{sb_t:.1f}), A/bg={A_t/bg:.2f}, "
          f"contour {g_r:.1f}σ → {ellipse.sum()} px")

# Remove overlap with base mask for clean accounting
layer4 &= ~base_mask
ct4 = int(layer4.sum())
layer_stats["4_azimuthal_residual_blob"] = {
    "pixels": ct4,
    "pct_of_image": round(ct4 / (H * W) * 100, 3),
    "components_kept": len(kept_components),
    "refinement": refined_info,
}
print(f"Layer 4 (azimuthal-residual blob): {ct4} px ({ct4/(H*W)*100:.3f}%)")

# =========================================================================
# COMBINE & SAVE
# =========================================================================
mask = layer0 | layer1 | layer2 | layer3 | layer4
total = int(mask.sum())
in_panel = (img != 0) | (mask & (img == 0))  # real + gap pixels that got masked
panel_frac = mask[img != 0].mean() if (img != 0).any() else 0.0

print(f"\n=== TOTAL MASK: {total} px ({total/(H*W)*100:.2f}% of image, "
      f"{panel_frac*100:.2f}% of panel area) ===")

# Save mask
np.save(f"{OUT}/mask_assembled.npy", mask)
print(f"Saved {OUT}/mask_assembled.npy")

# Save layer stats
layer_stats["_total"] = {
    "pixels": total,
    "pct_of_image": round(total / (H * W) * 100, 3),
    "pct_of_panels": round(panel_frac * 100, 3),
}
with open(f"{OUT}/_mask_layers.json", "w") as f:
    json.dump(layer_stats, f, indent=2, default=str)
print(f"Saved {OUT}/_mask_layers.json")
