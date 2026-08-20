#!/usr/bin/env python3
"""Step 4 (deterministic executor): azimuthal I(q) profile + quality metrics.

Expects in outputs/<run>/:
  masked_sum.npy   (1064, 1030) float64, NaN = masked   — from step 3

Writes:
  iq.npy           structured array: r_px, q_invA, I_mean, I_med, n_valid
  iq.png           I(q) curve with expected-ring markers + background windows
  iq_metrics.json  ring/background/coverage metrics consumed by the verifier

Geometry (from skills/masking/02_pyfai_azimuthal_sigmaclip.md, verified against
the stored q-map): dist 190 mm, lambda 1.2915 A (9.6 keV), 75 um pixels.
The assembled-image center is loaded from image_center.json; it is never hidden
as a constant in this executor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIST_M = 0.190  # sample-detector distance
PIX_M = 75e-6  # pixel pitch
LAMBDA_A = 1.2915  # wavelength

EXPECTED_RINGS_PX = [847, 1245]  # indexed LaB6 (100)/(110) lattice rings only
PROTECTED_FEATURE_BANDS_PX = [367, 531, 734, 847, 1245]
# inner three are unindexed diffuse anchors,
# protected from artifact screening but never
# treated as LaB6 center constraints
RING_SEARCH_PX = 20  # peak search half-window around expected radius
RING_BG_OFFSET = (30, 70)  # sideband offsets used to estimate local background
BG_WINDOWS = [
    (150, 330),
    (410, 500),
    (570, 700),
    (770, 820),
    (880, 1000),
    (1150, 1200),
    (1300, 1380),
]
MIN_BIN_PIXELS = 50  # radial bins with fewer valid pixels are dropped


def r_to_q(r_px: np.ndarray) -> np.ndarray:
    two_theta = np.arctan(r_px * PIX_M / DIST_M)
    return 4 * np.pi / LAMBDA_A * np.sin(two_theta / 2)


def load_image_center(out: Path) -> tuple[float, float]:
    path = out / "image_center.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required; run the optional center gate before verification"
        )
    artifact = json.loads(path.read_text())
    if artifact.get("coordinate_system") != "assembled_row_col":
        raise ValueError(f"unsupported coordinate system in {path}")
    center = artifact.get("center") or {}
    row, col = float(center["row"]), float(center["col"])
    if not (np.isfinite(row) and np.isfinite(col)):
        raise ValueError(f"non-finite image center in {path}")
    return row, col


def radial_profile(img: np.ndarray, center: tuple[float, float]):
    H, W = img.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - center[0], cc - center[1])
    rbin = rad.astype(int)
    ok = np.isfinite(img)
    nb = rbin.max() + 1
    n = np.bincount(rbin[ok], minlength=nb)
    s = np.bincount(rbin[ok], weights=img[ok], minlength=nb)
    mean = np.where(n >= MIN_BIN_PIXELS, s / np.maximum(n, 1), np.nan)
    med = np.full(nb, np.nan)
    flat_bins = rbin[ok]
    flat_vals = img[ok]
    order = np.argsort(flat_bins, kind="stable")
    fb, fv = flat_bins[order], flat_vals[order]
    edges = np.searchsorted(fb, np.arange(nb + 1))
    for i in range(nb):
        if n[i] >= MIN_BIN_PIXELS:
            med[i] = np.median(fv[edges[i] : edges[i + 1]])
    return np.arange(nb), mean, med, n


def rolling_median(x: np.ndarray, w: int = 31) -> np.ndarray:
    out = np.full_like(x, np.nan)
    h = w // 2
    for i in range(len(x)):
        seg = x[max(0, i - h) : i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(5, w // 4):
            out[i] = np.median(seg)
    return out


def azimuthal_uniformity(img: np.ndarray, center: tuple[float, float]) -> dict:
    """I(θ) sector scan of the masked endpoint (skill-06 method, deterministic).

    destriped radial residual → (Δr=15px × Δθ=2°) cell medians → per-band θ-detrend
    → robust |z| → seeds |z|>4.5, grow |z|>2.5, clusters ≥2 cells, ring bands excluded.
    Any POSITIVE off-ring cluster means unmasked contamination in the endpoint.
    """
    from scipy import ndimage

    H, W = img.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - center[0], cc - center[1])
    rbin = rad.astype(int)
    ok = np.isfinite(img)
    med_r = np.full(rbin.max() + 1, np.nan)
    for i in np.unique(rbin[ok]):
        sel = ok & (rbin == i)
        if sel.sum() >= 40:
            med_r[i] = np.median(img[sel])
    v = np.isfinite(med_r)
    k3 = np.ones(3) / 3
    med_s = med_r.copy()
    med_s[v] = np.convolve(np.nan_to_num(med_r), k3, "same")[v] / np.maximum(
        np.convolve(v.astype(float), k3, "same")[v], 1e-9
    )
    resid = np.where(ok, img - med_s[rbin], np.nan)
    resid = resid - np.nanmedian(resid, axis=0)[None, :]
    resid = resid - np.nanmedian(resid, axis=1)[:, None]

    R0b, R1b, TH0, TH1, DR, DTH, MINPX = 120, 1400, -100.0, 10.0, 15, 2.0, 60
    th = np.degrees(np.arctan2(rr - center[0], cc - center[1]))
    nb, nsec = int((R1b - R0b) / DR), int((TH1 - TH0) / DTH)
    bi = ((rad - R0b) / DR).astype(int)
    si = ((th - TH0) / DTH).astype(int)
    inside = (rad >= R0b) & (rad < R1b) & (th >= TH0) & (th < TH1) & np.isfinite(resid)
    cid = bi * nsec + si
    order = np.argsort(cid[inside], kind="stable")
    fs, vs = cid[inside][order], resid[inside][order]
    edges = np.searchsorted(fs, np.arange(nb * nsec + 1))
    med = np.full((nb, nsec), np.nan)
    for c in range(nb * nsec):
        a, b = edges[c], edges[c + 1]
        if b - a >= MINPX:
            med[c // nsec, c % nsec] = np.median(vs[a:b])
    det = np.full_like(med, np.nan)
    h = 15 // 2
    for i in range(nb):
        for j in range(nsec):
            seg = med[i, max(0, j - h) : j + h + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 5 and np.isfinite(med[i, j]):
                det[i, j] = med[i, j] - np.median(seg)
    gm = np.nanmedian(det)
    sig = 1.4826 * np.nanmedian(np.abs(det - gm))
    z = (det - gm) / max(sig, 1e-9)
    ring_band = np.array(
        [
            any(abs(R0b + (i + 0.5) * DR - rp) <= 25 for rp in PROTECTED_FEATURE_BANDS_PX)
            for i in range(nb)
        ]
    )
    zf = np.nan_to_num(np.abs(z))
    seeds = (zf > 4.5) & ~ring_band[:, None]
    grown = ndimage.binary_propagation(seeds, mask=(zf > 2.5) & ~ring_band[:, None])
    lab, ncl = ndimage.label(grown)
    clusters = []
    for k in range(1, ncl + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < 2:
            continue
        clusters.append(
            {
                "cells": int(len(ys)),
                "r_px": round(float(R0b + (ys.mean() + 0.5) * DR), 1),
                "theta_deg": round(float(TH0 + (xs.mean() + 0.5) * DTH), 1),
                "max_abs_z": round(float(zf[ys, xs].max()), 2),
                "sign": int(np.sign(np.nanmedian(det[ys, xs]))),
            }
        )
    return {
        "grid": {
            "dr_px": DR,
            "dtheta_deg": DTH,
            "min_px": MINPX,
            "z_seed": 4.5,
            "z_grow": 2.5,
            "min_cells": 2,
            "ring_margin_px": 25,
        },
        "cell_sigma_adu": round(float(sig), 2),
        "n_offring_clusters": len(clusters),
        "n_positive": sum(1 for c in clusters if c["sign"] > 0),
        "n_negative": sum(1 for c in clusters if c["sign"] < 0),
        "max_offring_abs_z": max((c["max_abs_z"] for c in clusters), default=0.0),
        "clusters": clusters,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    img = np.load(out / "masked_sum.npy")
    center = load_image_center(out)
    r, I_mean, I_med, n = radial_profile(img, center)
    q = r_to_q(r.astype(float))

    arr = np.zeros(
        len(r),
        dtype=[
            ("r_px", "f8"),
            ("q_invA", "f8"),
            ("I_mean", "f8"),
            ("I_med", "f8"),
            ("n_valid", "i8"),
        ],
    )
    arr["r_px"], arr["q_invA"], arr["I_mean"], arr["I_med"], arr["n_valid"] = r, q, I_mean, I_med, n
    np.save(out / "iq.npy", arr)

    # ---- ring metrics -----------------------------------------------------
    rings = []
    for exp in EXPECTED_RINGS_PX:
        lo, hi = exp - RING_SEARCH_PX, exp + RING_SEARCH_PX
        win = I_mean[lo:hi]
        if not np.isfinite(win).any():
            rings.append({"expected_px": exp, "found": False})
            continue
        k = int(np.nanargmax(win))
        pk_r, pk_i = lo + k, float(win[k])
        sb = []
        for s0, s1 in [
            (exp - RING_BG_OFFSET[1], exp - RING_BG_OFFSET[0]),
            (exp + RING_BG_OFFSET[0], exp + RING_BG_OFFSET[1]),
        ]:
            seg = I_mean[max(0, s0) : s1]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                sb.append(np.median(seg))
        bg = float(np.mean(sb)) if sb else np.nan
        contrast = (pk_i - bg) / bg if np.isfinite(bg) and bg > 0 else np.nan
        rings.append(
            {
                "expected_px": exp,
                "found": True,
                "found_px": int(pk_r),
                "delta_px": int(pk_r - exp),
                "I_peak": pk_i,
                "I_bg_local": bg,
                "contrast": round(float(contrast), 4),
            }
        )

    # ---- background metrics ----------------------------------------------
    bg_bins, bg_vals = [], []
    for w0, w1 in BG_WINDOWS:
        seg = I_mean[w0:w1]
        m = np.isfinite(seg)
        bg_bins.extend((np.arange(w0, w1)[m]).tolist())
        bg_vals.extend(seg[m].tolist())
    bg_vals = np.array(bg_vals)
    trend = rolling_median(bg_vals, 31)
    resid = bg_vals - trend
    resid = resid[np.isfinite(resid)]
    mad = float(np.median(np.abs(resid - np.median(resid))) * 1.4826) if len(resid) else np.nan
    level = float(np.nanmedian(bg_vals))
    background = {
        "windows_px": BG_WINDOWS,
        "n_bins": int(len(bg_vals)),
        "neg_bin_fraction": round(float((bg_vals < 0).mean()), 5) if len(bg_vals) else None,
        "rel_noise": round(mad / level, 5) if level > 0 else None,
        "bump_max_sigma": (
            round(float(np.max(np.abs(resid)) / mad), 2) if len(resid) and mad > 0 else None
        ),
        "median_level": round(level, 2),
    }

    # ---- coverage ---------------------------------------------------------
    finite = np.isfinite(img)
    coverage = {
        "unmasked_pixels": int(finite.sum()),
        "valid_bins": int(np.isfinite(I_mean).sum()),
        "bins_total": int(len(r)),
    }
    log3 = out / "mask_log.json"
    if log3.exists():
        coverage["panel_mask_fraction"] = json.loads(log3.read_text()).get(
            "mask_fraction_of_panels"
        )

    metrics = {
        "image_center": list(center),
        "image_center_artifact": "image_center.json",
        "geometry": {"dist_m": DIST_M, "pixel_m": PIX_M, "lambda_A": LAMBDA_A},
        "expected_rings_px": EXPECTED_RINGS_PX,
        "rings": rings,
        "background": background,
        "coverage": coverage,
        "azimuthal_uniformity": azimuthal_uniformity(img, center),
    }
    (out / "iq_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- plot -------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(r, I_mean, lw=1, label="I(r) azimuthal mean")
    ax.plot(r, I_med, lw=0.8, alpha=0.6, label="I(r) azimuthal median")
    for exp in EXPECTED_RINGS_PX:
        ax.axvline(exp, color="gray", ls=":", alpha=0.7)
    for w0, w1 in BG_WINDOWS:
        ax.axvspan(w0, w1, color="#f5a623", alpha=0.08)
    ax.set_xlabel("radius from image center (px)   [top axis: q]")
    ax.set_ylabel("mean summed intensity (ADU)")
    # Full valid range — NEVER hard-cap the axis (a fixed 1100 px cap hid the
    # outer 1245 px ring, q>2). Upper limit = last bin with enough statistics.
    r_valid_max = (
        float(np.max(r[np.isfinite(I_mean)])) if np.isfinite(I_mean).any() else float(r[-1])
    )
    ax.set_xlim(0, r_valid_max + 25)

    def q_to_r(qq):
        angle = 2 * np.arcsin(np.asarray(qq, float) * LAMBDA_A / (4 * np.pi))
        return np.tan(angle) * DIST_M / PIX_M

    top = ax.secondary_xaxis("top", functions=(lambda x: r_to_q(np.asarray(x, float)), q_to_r))
    top.set_xlabel("q (1/A)")
    ax.legend()
    ax.set_title(
        f"{out.name}: I(q) — rings expected at {EXPECTED_RINGS_PX} px (dotted), bg windows shaded"
    )
    fig.tight_layout()
    fig.savefig(out / "iq.png", dpi=110)

    print(f"[done] iq.npy / iq.png / iq_metrics.json in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
