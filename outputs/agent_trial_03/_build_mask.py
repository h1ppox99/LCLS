#!/usr/bin/env python3
"""Build the detector mask for agent_trial_03 (Jungfrau1M, LaB6, Run 475).

Layers:
  0  zero_dead      — exactly-zero pixels (panel gaps) dilated 1 px + all negative pixels
  1  status_bad     — calib/status_bad.npy mapped to assembled space via ix/iy
  2  beamstop       — horizontal shadow band (~rows 970-1006) + direct-beam disc
  3  asic_boundary  — double-width edge rows/cols flanking inter-module and inter-ASIC gaps
  4  azimuthal_blob — method-05 azimuthal-residual parasitic blob detection
"""
import numpy as np
from scipy import ndimage

OUT = "outputs/agent_trial_03"
CAL = "calib"

# ── Load data ─────────────────────────────────────────────────────────
img = np.load(f"{OUT}/sum_assembled.npy")    # (1064, 1030) float64
H, W = img.shape
assert (H, W) == (1064, 1030), f"unexpected shape {img.shape}"

status_bad = np.load(f"{CAL}/status_bad.npy")  # (2,512,1024) bool
ix = np.load(f"{CAL}/ix.npy")                  # (2,512,1024) uint16
iy = np.load(f"{CAL}/iy.npy")                  # (2,512,1024) uint16

layers = {}   # name → (bool array of NEW pixels, raw count including overlaps)
cumulative = np.zeros((H, W), dtype=bool)

def add_layer(name, raw_mask):
    """Register a new layer: only count pixels not already masked."""
    new = raw_mask & ~cumulative
    layers[name] = (new, int(raw_mask.sum()))
    cumulative[:] |= raw_mask
    n = int(new.sum())
    pct = n / (H * W) * 100
    print(f"  layer '{name}': {n:>7d} new px  ({pct:.3f}%),  raw={int(raw_mask.sum())}")
    return new

# ═══ Layer 0: zero/dead mask + gap-edge dilation ═════════════════════
# Exactly-zero pixels are panel gaps / geometry holes.
# Dilate these by 1 px (3×3 square) to mask unreliable gap-edge pixels.
# Negative pixels are dead/bad — include directly without dilation.
zeros = (img == 0)
struct = np.ones((3, 3), dtype=bool)
zero_dilated = ndimage.binary_dilation(zeros, structure=struct, iterations=1)
negatives = (img < 0)
dead_mask = zero_dilated | negatives
add_layer("zero_dead_dilated", dead_mask)

# ═══ Layer 1: status_bad from calibration ═════════════════════════════
sb_assembled = np.zeros((H, W), dtype=bool)
sb_assembled[ix[status_bad], iy[status_bad]] = True
add_layer("status_bad", sb_assembled)

# ═══ Layer 2: beamstop shadow band ════════════════════════════════════
# Beam center at (row 992, col 35). Horizontal shadow band identified by
# row-median analysis: normal rows have median ~300-400, shadow rows drop
# to ~80-130. The band spans ~rows 970-1006 across full detector width.
#
# Evidence:
#   row 969: median 322 (normal)
#   row 974: median 190 (entering shadow)
#   row 977-1001: median 80-130 (deep shadow)
#   row 1004: median 213 (exiting)
#   row 1007: median 329 (normal)
#
# Shadow is deepest far from beam center (cols 800+: ratio ~0.16 vs normal)
# and shallower near beam center (cols 0-100: ratio ~0.57).

bs_mask = np.zeros((H, W), dtype=bool)

# 2a: Horizontal shadow band — rows 970-1006 (task-specified bounds, confirmed
# by row-median dropping below 50% of normal in this range)
bs_mask[970:1007, :] = True   # rows 970 through 1006 inclusive

# 2b: Direct beam vicinity — 15-px radius circle around (992, 35)
# Catches extreme-value pixels at the beamstop attachment point and
# direct-beam spill/scatter that leaks around the beamstop edge.
bc_row, bc_col = 992.0, 35.0
rr, cc = np.ogrid[0:H, 0:W]
beam_disc = ((rr - bc_row)**2 + (cc - bc_col)**2) <= 15**2
bs_mask |= beam_disc

add_layer("beamstop_shadow", bs_mask)

# ═══ Layer 3: ASIC boundary double-width edges ═══════════════════════
# The Jungfrau1M has 2 panels × 2×4 ASICs (256×256 each). In assembled
# space the raw detector's ASIC edges become double-width pixels that
# collect ~1.4-2.2× the signal of normal pixels. Verified by median ratio:
#   row 255: 2.14×, row 258: 2.19×, row 513: 1.64×, row 550: 1.38×,
#   row 805: 1.84×, row 808: 1.84×.
#
# From ix/iy mapping:
#   Edge ROWS: 0, 255, 258, 513, 550, 805, 808, 1063
#   Edge COLS: 0, 255, 258, 513, 516, 771, 774, 1029
#   Gap ROWS (already zero): 256-257, 514-549, 806-807
#   Gap COLS (already zero): 256-257, 514-515, 772-773

asic_mask = np.zeros((H, W), dtype=bool)

edge_rows = [0, 255, 258, 513, 550, 805, 808, 1063]
for r in edge_rows:
    asic_mask[r, :] = True

edge_cols = [0, 255, 258, 513, 516, 771, 774, 1029]
for c in edge_cols:
    asic_mask[:, c] = True

add_layer("asic_boundary", asic_mask)

# ═══ Layer 4: azimuthal-residual blob detection (method 05) ═══════════
# Detects diffuse parasitic-scattering blobs that are too low-contrast
# for per-pixel sigma-clipping. Uses radial-median model, destriped
# residual, matched-scale Gaussian smoothing, and hysteresis growth.
# Reference: skills/masking/05_azimuthal_residual.md
# Known target: parasitic blob at (row 686, col 387), radius ~466 px.

def azimuthal_residual_layer(img, base_mask, bc=(992.0, 35.0),
                             sigma_smooth=8.0, z_seed=5.0, z_grow=3.0,
                             min_size=150, ring_radii=(367, 531, 734, 847, 1245),
                             ring_margin=60, off_ring_only=True, r_min=100.0):
    safe = np.where(base_mask | (np.abs(img) > 1e5), np.nan, img)
    H, W = img.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - bc[0], cc - bc[1]); rbin = rad.astype(int)
    ok = ~np.isnan(safe)
    med_r = np.full(rbin.max() + 1, np.nan)
    for i in np.unique(rbin[ok]):
        sel = ok & (rbin == i)
        if sel.sum() >= 40: med_r[i] = np.median(safe[sel])
    v = ~np.isnan(med_r); k = np.ones(3) / 3
    med_s = med_r.copy()
    med_s[v] = np.convolve(np.nan_to_num(med_r), k, "same")[v] / \
               np.maximum(np.convolve(v.astype(float), k, "same")[v], 1e-9)
    resid = safe - med_s[rbin]
    resid -= np.nanmedian(resid, axis=0)[None, :]          # destripe cols
    resid -= np.nanmedian(resid, axis=1)[:, None]          # destripe rows
    ok2 = ~np.isnan(resid)
    sm = ndimage.gaussian_filter(np.nan_to_num(resid), sigma_smooth)
    w = ndimage.gaussian_filter(ok2.astype(float), sigma_smooth)
    sm = np.where(w > 0.6, sm / np.maximum(w, 1e-9), np.nan)
    zmap = (sm - np.nanmedian(sm)) / (1.4826 * np.nanmedian(np.abs(sm - np.nanmedian(sm))))
    zf = np.nan_to_num(zmap)
    lab, n = ndimage.label(zf > z_seed)
    sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
    seeds = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_size])
    grown = ndimage.binary_propagation(seeds, mask=zf > z_grow)
    layer = np.zeros_like(base_mask)
    glab, gn = ndimage.label(grown)
    print(f"    [azimuthal] found {gn} components in z > z_seed:")
    for g in range(1, gn + 1):
        comp = glab == g
        r_comp = rad[comp].mean()
        area = int(comp.sum())
        peak_z = float(zf[comp].max())
        rows_c, cols_c = np.where(comp)
        centroid = (float(rows_c.mean()), float(cols_c.mean()))
        on_ring = any(abs(r_comp - rp) <= ring_margin for rp in ring_radii)
        skip_near = r_comp < r_min
        keep = (not skip_near) and ((not off_ring_only) or (not on_ring))
        status = "KEEP" if keep else ("skip:near-beam" if skip_near else "skip:on-ring")
        print(f"      comp {g}: centroid=({centroid[0]:.0f},{centroid[1]:.0f}), r={r_comp:.0f}, "
              f"area={area}, peak_z={peak_z:.1f} → {status}")
        if keep:
            layer |= comp
    layer &= ~base_mask
    return layer, zmap

blob_layer, zmap = azimuthal_residual_layer(img, cumulative)
add_layer("azimuthal_blob", blob_layer)

# ═══ Save final mask ══════════════════════════════════════════════════
np.save(f"{OUT}/mask_assembled.npy", cumulative)

# In-panel area (non-gap pixels)
in_panel = img != 0  # original non-gap pixels
panel_masked = cumulative[in_panel].sum()
panel_total = in_panel.sum()

print(f"\n{'='*60}")
print(f"  TOTAL masked: {cumulative.sum():,} px ({cumulative.mean()*100:.2f}% of image)")
print(f"  Panel-only:   {panel_masked:,} / {panel_total:,} ({panel_masked/panel_total*100:.2f}%)")
print(f"  Saved → {OUT}/mask_assembled.npy  shape={cumulative.shape} dtype={cumulative.dtype}")

# Save per-layer metadata for rationale
import json
layer_meta = {}
for name, (new_px, raw_count) in layers.items():
    layer_meta[name] = {
        "new_pixels": int(new_px.sum()),
        "raw_pixels": raw_count,
        "new_pct_of_image": float(new_px.sum() / (H * W) * 100),
    }
layer_meta["TOTAL"] = {
    "pixels": int(cumulative.sum()),
    "pct_of_image": float(cumulative.mean() * 100),
    "pct_of_panel": float(panel_masked / panel_total * 100),
}
with open(f"{OUT}/_mask_layer_counts.json", "w") as f:
    json.dump(layer_meta, f, indent=2)
print(f"  Layer counts → {OUT}/_mask_layer_counts.json")
