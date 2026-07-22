#!/usr/bin/env python3
"""Build the multi-layer assembled-frame mask for agent_trial_05.

Layers
------
0. Zero/dead + dilation    — pixels == 0 (panel gaps / border) plus extreme outliers
                             (|value| > 1e5), dilated by 1 px to exclude unreliable
                             gap-edge and ASIC-boundary double-width pixels.
1. Status-bad (calib)      — calib/status_bad.npy mapped to assembled coords via ix/iy.
2. Beamstop shadow band    — horizontal shadow rows 968-1008 (penumbra included,
                             <90% of expected intensity) + beam-center circle r<25 px.
3. (ASIC boundary)         — absorbed into layer 0 dilation; see rationale.
4. Azimuthal-residual blob — parasitic blob at (row ~686, col ~387) detected via
                             method 05 (radial-median model, destripe, smooth, threshold).

Output: mask_assembled.npy  (1064, 1030) bool, True = masked.
"""

import numpy as np
from scipy import ndimage
from pathlib import Path

OUT = Path("outputs/agent_trial_05")
CAL = Path("calib")

img = np.load(OUT / "sum_assembled.npy")
H, W = img.shape  # (1064, 1030)
print(f"Image shape: {H} x {W}, total pixels: {H*W}")

layer_counts = {}  # track NEW pixels per layer (non-overlapping)

# ── Layer 0: Zero / dead / extreme + dilation ──────────────────────────
zero_mask = (img == 0)
extreme_mask = (np.abs(img) > 1e5)    # extreme negative or positive outliers
base = zero_mask | extreme_mask
# Dilate by 1 pixel (structuring element = 3x3 cross) to catch:
#   a) unreliable gap-edge pixels adjacent to panel boundaries
#   b) ASIC-boundary double-width pixels (systematically ~2× intensity)
layer0 = ndimage.binary_dilation(base, iterations=1)
layer_counts["L0_zero_dead_dilation"] = int(layer0.sum())
print(f"Layer 0 (zero/dead+extreme+dilation): {layer0.sum()} px "
      f"({layer0.mean()*100:.2f}%)")
print(f"  - raw zeros:       {zero_mask.sum()}")
print(f"  - extreme (|v|>1e5): {extreme_mask.sum()}")
print(f"  - after dilation:  {layer0.sum()}")

cumul = layer0.copy()

# ── Layer 1: Status-bad from calibration ───────────────────────────────
status_bad = np.load(CAL / "status_bad.npy")   # (2, 512, 1024) bool
ix = np.load(CAL / "ix.npy")                   # (2, 512, 1024) uint16  -> row in assembled
iy = np.load(CAL / "iy.npy")                   # (2, 512, 1024) uint16  -> col in assembled

# Map bad pixels to assembled coordinates
layer1_raw = np.zeros((H, W), dtype=bool)
bad_mask = status_bad.astype(bool)
if bad_mask.ndim == 3 and bad_mask.shape[0] > 1:
    # Collapse over gain stages if present
    bad_any = bad_mask.any(axis=0) if bad_mask.shape[0] == 3 else bad_mask
    # If shape is (2, 512, 1024), it's 2 panels, not gain stages
    if bad_mask.shape[0] == 2 and bad_mask.shape[1] == 512:
        bad_any = bad_mask  # keep panel dimension
else:
    bad_any = bad_mask

# Map each bad pixel to assembled coords
for p in range(bad_any.shape[0]):
    bad_idx = np.where(bad_any[p])
    rows_asm = ix[p][bad_idx]
    cols_asm = iy[p][bad_idx]
    valid = (rows_asm < H) & (cols_asm < W)
    layer1_raw[rows_asm[valid], cols_asm[valid]] = True

layer1 = layer1_raw & ~cumul  # new pixels only
layer_counts["L1_status_bad"] = int(layer1.sum())
cumul |= layer1
print(f"Layer 1 (status-bad calib): {layer1.sum()} new px "
      f"(total bad mapped: {layer1_raw.sum()})")

# ── Layer 2: Beamstop shadow band + beam center ───────────────────────
# Shadow band: rows 968-1008 where row median < 90% of reference
# (validated: ref_above ~404, ref_below ~385, avg ~395; shadow rows show 83-345)
bc_row, bc_col = 992.0, 35.0

# Horizontal shadow band
layer2 = np.zeros((H, W), dtype=bool)
layer2[968:1009, :] = True  # rows 968-1008 inclusive

# Add circular beamstop area near beam center (radius < 25 px)
rr, cc = np.mgrid[0:H, 0:W]
rad = np.hypot(rr - bc_row, cc - bc_col)
layer2 |= (rad < 25)

# Don't mask gap pixels (they're already zero and handled by L0)
layer2 &= (img != 0)  # only mask real pixels in the shadow, not gap cells
layer2_new = layer2 & ~cumul
layer_counts["L2_beamstop_shadow"] = int(layer2_new.sum())
cumul |= layer2_new
print(f"Layer 2 (beamstop shadow band + beam center): {layer2_new.sum()} new px")

# ── Layer 3: ASIC boundary ────────────────────────────────────────────
# The gap-adjacent double-width ASIC boundary pixels (rows 255,258,513,550,805,808;
# cols 255,258,513,516,771,774) are systematically ~2× the normal intensity.
# They are ALREADY captured by Layer 0's 1-px dilation of the zero-gap mask.
# Verified: row 255 median=646 vs normal ~300 (gap-adj), row 258 median=640.
# No additional pixels needed — this layer adds 0 new pixels.
layer3_new_count = 0
layer_counts["L3_asic_boundary"] = layer3_new_count
print(f"Layer 3 (ASIC boundary): {layer3_new_count} new px "
      f"(absorbed into L0 dilation)")

# ── Layer 4: Azimuthal-residual blob detection (method 05) ────────────
# Parasitic scatter blob at (row ~686, col ~387), r ≈ 466 from beam center,
# +59% excess over azimuthal median at that radius.

def azimuthal_residual_layer(img, base_mask, bc=(992.0, 35.0),
                             sigma_smooth=8.0, z_seed=5.0, z_grow=3.0,
                             min_size=150, ring_radii=(367, 531, 734, 847, 1245),
                             ring_margin=60, off_ring_only=True, r_min=100.0,
                             refine_contour_eps=0.10):
    """Method 05: azimuthal-residual blob detection with parametric refinement."""
    safe = np.where(base_mask | (np.abs(img) > 1e5), np.nan, img)
    H, W = img.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - bc[0], cc - bc[1])
    rbin = rad.astype(int)
    ok = ~np.isnan(safe)

    # Azimuthal median model
    med_r = np.full(rbin.max() + 1, np.nan)
    for i in np.unique(rbin[ok]):
        sel = ok & (rbin == i)
        if sel.sum() >= 40:
            med_r[i] = np.median(safe[sel])
    v = ~np.isnan(med_r)
    k = np.ones(3) / 3
    med_s = med_r.copy()
    med_s[v] = (np.convolve(np.nan_to_num(med_r), k, "same")[v] /
                np.maximum(np.convolve(v.astype(float), k, "same")[v], 1e-9))

    resid = safe - med_s[rbin]
    # Destripe (CRITICAL per method 05 — reduces noise 69→31 ADU)
    resid -= np.nanmedian(resid, axis=0)[None, :]   # column stripes
    resid -= np.nanmedian(resid, axis=1)[:, None]    # row stripes

    ok = ~np.isnan(resid)
    sm = ndimage.gaussian_filter(np.nan_to_num(resid), sigma_smooth)
    w = ndimage.gaussian_filter(ok.astype(float), sigma_smooth)
    sm = np.where(w > 0.6, sm / np.maximum(w, 1e-9), np.nan)

    zmap = (sm - np.nanmedian(sm)) / (1.4826 * np.nanmedian(np.abs(sm - np.nanmedian(sm))))
    zf = np.nan_to_num(zmap)

    # Seed detection
    lab, n = ndimage.label(zf > z_seed)
    sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
    seeds = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_size])
    grown = ndimage.binary_propagation(seeds, mask=zf > z_grow)

    layer = np.zeros_like(base_mask)
    glab, gn = ndimage.label(grown)
    kept = []
    for g in range(1, gn + 1):
        comp = glab == g
        r_comp = rad[comp].mean()
        if r_comp < r_min:
            continue
        on_ring = any(abs(r_comp - rp) <= ring_margin for rp in ring_radii)
        if (not off_ring_only) or (not on_ring):
            kept.append(comp)
            layer |= comp
            print(f"  Blob component: r_mean={r_comp:.0f}, size={comp.sum()} px, "
                  f"z_peak={zf[comp].max():.1f}, on_ring={on_ring}")

    # Parametric refinement (contamination contour)
    if refine_contour_eps and kept:
        from scipy.optimize import least_squares
        for comp in kept:
            ys, xs = np.where(comp)
            y0g, x0g = float(ys.mean()), float(xs.mean())
            r_eq = np.sqrt(comp.sum() / np.pi)
            half = int(np.clip(2.5 * r_eq + 2 * sigma_smooth, 60, 300))
            sl = (slice(max(0, int(y0g) - half), min(H, int(y0g) + half)),
                  slice(max(0, int(x0g) - half), min(W, int(x0g) + half)))
            Zc, Yc, Xc = sm[sl], rr[sl], cc[sl]
            good = np.isfinite(Zc)
            if good.sum() < 500:
                continue
            def gmod(p, x, y):
                A, x0, y0, sa, sb, off = p
                return A * np.exp(-(x - x0)**2 / (2 * sa**2)
                                  - (y - y0)**2 / (2 * sb**2)) + off
            s0 = float(np.clip(r_eq, 1.2 * sigma_smooth, 100.0))
            try:
                res = least_squares(
                    lambda p: gmod(p, Xc[good], Yc[good]) - Zc[good],
                    [float(np.nanmax(Zc)), x0g, y0g, s0, s0, 0.0],
                    loss="soft_l1", f_scale=30)
                p = res.x
            except Exception:
                continue
            A_f, x0, y0, sa_f, sb_f, _ = p
            # Deconvolve the smoothing kernel
            sa_t = np.sqrt(max(sa_f**2 - sigma_smooth**2, 9.0))
            sb_t = np.sqrt(max(sb_f**2 - sigma_smooth**2, 9.0))
            A_t = A_f * (abs(sa_f) * abs(sb_f)) / (sa_t * sb_t)
            bg = med_s[min(int(np.hypot(y0 - bc[0], x0 - bc[1])), len(med_s) - 1)]
            if not (np.isfinite(bg) and bg > 0 and A_t > refine_contour_eps * bg):
                continue
            g_r = np.sqrt(2 * np.log(A_t / (refine_contour_eps * bg)))
            ellipse = (((cc - x0) / (sa_t * g_r))**2
                       + ((rr - y0) / (sb_t * g_r))**2) <= 1
            n_before = layer.sum()
            layer |= ellipse
            print(f"  Refinement: center=({y0:.0f},{x0:.0f}), "
                  f"sigma=({sa_t:.1f},{sb_t:.1f}), "
                  f"A_t/bg={A_t/bg:.1%}, "
                  f"contour at eps={refine_contour_eps}, "
                  f"added {layer.sum()-n_before} px")

    layer &= ~base_mask  # keep layers disjoint
    return layer, zmap

print("\nRunning azimuthal-residual blob detection...")
layer4, zmap = azimuthal_residual_layer(img, cumul)
layer_counts["L4_azimuthal_residual_blob"] = int(layer4.sum())
cumul |= layer4
print(f"Layer 4 (azimuthal-residual blob): {layer4.sum()} new px")

# ── Combine and save ──────────────────────────────────────────────────
mask = cumul.copy()
print(f"\n=== FINAL MASK ===")
print(f"Total masked: {mask.sum()} / {H*W} px ({mask.mean()*100:.2f}%)")

# Panel-only stats (exclude gap cells)
in_panel = (img != 0) | layer1_raw  # a pixel is "real" if it's non-zero or flagged bad
panel_count = in_panel.sum()
masked_panel = mask[in_panel].sum()
print(f"Panel-only: {masked_panel} / {panel_count} px ({masked_panel/panel_count*100:.2f}%)")

print(f"\nPer-layer pixel counts (new pixels each layer adds):")
for name, count in layer_counts.items():
    print(f"  {name}: {count}")

np.save(OUT / "mask_assembled.npy", mask)
print(f"\nSaved: {OUT / 'mask_assembled.npy'}")
print(f"  shape: {mask.shape}, dtype: {mask.dtype}")
