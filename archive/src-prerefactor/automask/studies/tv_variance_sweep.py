#!/usr/bin/env python3
"""
tv_variance_sweep.py -- Option 2 (Total-Variation regularization) for replacing
the close_open post-processing of the variance mask.

Idea:  the raw variance detector gives a continuous low-variance score field
z(p) = -MAD_z(log10 uSTD)  (high == wants masking).  Instead of thresholding it
and cleaning with morphology, we DENOISE the score with Total Variation first:

    u = argmin_u  ||u - z||^2 + weight * TV(u)          (Chambolle, isotropic)

then threshold at the SAME robust cut K the raw detector uses.  Because the TV
data term keeps u on z's scale, weight=0 reproduces the raw (z>K) mask exactly;
raising weight produces piecewise-constant regions with sharp straight edges --
it FILLS the beam-stop shadow from sparse hits and DROPS isolated weak specks,
all under one knob.  A pixel with strong evidence (very low variance -> large z)
survives even in isolation; a weak straggler loses to its neighbourhood.  `weight`
is the data-fidelity <-> shape-prior control parameter.

This script sweeps `weight` for BOTH run 475 and 389 and produces, per run:
  * a grid of TP/FP/FN agreement maps at increasing weight (robustness by eye)
  * metric curves  IoU / T-prec / T-rec  vs weight, with the close_open baseline
  * a 2-D IoU heatmap over (weight x threshold K) -- robustness to BOTH knobs
Reads the frozen substrate written by shape_priors_variance.py is NOT required;
we recompute the full-frame score from ustd so TV sees real spatial context.

Run:  python tv_variance_sweep.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
from dataset import score                               # noqa: E402
from methods import geometry_mask, close_open, load_all  # noqa: E402

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
OUT_DIR = os.path.join(AUTOMASK, "outputs", "masks", "tv_variance")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = (475, 389)
# Two thresholds. KRAW is the raw cut method_variance/close_open use before
# morphology (kept so the baseline matches methods.py exactly). KOP is the TV
# operating threshold: the TV data term keeps u on z's scale but flattens the
# sparse peaks into a broad low plateau, so the mask must be read off at a LOWER
# cut -- see the study docstring / the (weight,K) sweep that motivated KOP=3.5.
KRAW = 7.0
KOP = 3.5
# weights shown in the agreement grid (0 == raw z>KOP, no smoothing)
GRID_W = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
# denser weight axis for the metric curves
CURVE_W = np.r_[0.0, np.geomspace(0.05, 16.0, 24)]
# threshold axis for the 2-D robustness heatmap (spans both runs' optima)
HEAT_K = np.linspace(2.5, 7.0, 13)


def score_field(ustd):
    """Full-frame low-variance score: high == strongly wants masking.
    Robust-MAD z of log10(uSTD), negated. Undefined pixels (uSTD==0) -> 0
    (neutral background) so TV sees no spurious gradient there."""
    m = ustd > 0
    z = np.zeros(ustd.shape)
    logr = np.log10(ustd[m])
    med = np.median(logr)
    mad = np.median(np.abs(logr - med)) * 1.4826 + 1e-12
    z[m] = -(logr - med) / mad
    return z


def agree_rgb(pred, truth):
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)      # TP green
    a[pred & ~truth] = (0.9, 0.0, 0.0)     # FP red
    a[~pred & truth] = (0.0, 0.3, 1.0)     # FN blue
    return a


def prep(run):
    sumimg, mean, ustd, human = load_all(run)
    real = sumimg != 0
    G = geometry_mask(real)
    cand = real & ~G
    target = human & ~G
    z = score_field(ustd)
    return dict(z=z, cand=cand, target=target, real=real, G=G)


def tv_mask(z, cand, weight, k=KOP):
    if weight <= 0:
        u = z
    else:
        u = denoise_tv_chambolle(z, weight=weight)
    return (u > k) & cand


def main():
    stash = {}
    for run in RUNS:
        stash[run] = prep(run)

    # ---- baselines (close_open) per run --------------------------------------
    base = {}
    for run in RUNS:
        s = stash[run]
        raw = (s["z"] > KRAW) & s["cand"]
        clean = close_open(raw) & s["real"]
        base[run] = score(clean, s["target"])
        print(f"run {run}: close_open baseline  IoU {base[run]['iou']:.3f}  "
              f"prec {base[run]['precision']:.3f}  rec {base[run]['recall']:.3f}")

    # ---- agreement grid per run ----------------------------------------------
    for run in RUNS:
        s = stash[run]
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for a, w in zip(ax.ravel(), GRID_W):
            M = tv_mask(s["z"], s["cand"], w)
            sc = score(M, s["target"])
            a.imshow(agree_rgb(M, s["target"]))
            tag = "raw (weight=0)" if w == 0 else f"weight={w:g}"
            a.set_title(f"{tag}\nIoU={sc['iou']:.3f} prec={sc['precision']:.2f} "
                        f"rec={sc['recall']:.2f}", fontsize=11)
            a.axis("off")
        fig.suptitle(f"xppl1016922 run {run} -- TV-denoised variance mask vs weight "
                     f"(green=TP red=FP blue=FN; threshold K={KOP}, close_open IoU="
                     f"{base[run]['iou']:.3f})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(FIG_DIR, f"tv_variance_grid_run{run:04d}.png")
        fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"  [saved] {out}")

    # ---- metric curves vs weight (both runs) ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, run in zip(axes, RUNS):
        s = stash[run]
        iou, prec, rec = [], [], []
        for w in CURVE_W:
            sc = score(tv_mask(s["z"], s["cand"], w), s["target"])
            iou.append(sc["iou"]); prec.append(sc["precision"]); rec.append(sc["recall"])
        ax.plot(CURVE_W, iou, "-o", ms=3, label="IoU", color="C0")
        ax.plot(CURVE_W, prec, "-s", ms=3, label="T-precision", color="C1")
        ax.plot(CURVE_W, rec, "-^", ms=3, label="T-recall", color="C2")
        ax.axhline(base[run]["iou"], ls="--", color="C0", alpha=0.6,
                   label=f"close_open IoU ({base[run]['iou']:.3f})")
        best = int(np.argmax(iou))
        ax.axvline(CURVE_W[best], ls=":", color="k", alpha=0.5)
        ax.set_title(f"run {run}  --  best IoU {iou[best]:.3f} at weight={CURVE_W[best]:.3g}")
        ax.set_xlabel("TV weight  (0 = raw, larger = stronger shape prior)")
        ax.set_ylabel("score"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="center right")
    fig.suptitle(f"TV-regularized variance mask: metrics vs weight (threshold K={KOP})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, "tv_variance_metrics.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # ---- 2-D IoU robustness heatmap (weight x threshold) ---------------------
    HEAT_W = np.r_[0.0, np.geomspace(0.05, 16.0, 15)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, run in zip(axes, RUNS):
        s = stash[run]
        H = np.zeros((len(HEAT_W), len(HEAT_K)))
        for i, w in enumerate(HEAT_W):
            u = s["z"] if w <= 0 else denoise_tv_chambolle(s["z"], weight=w)
            for j, k in enumerate(HEAT_K):
                H[i, j] = score((u > k) & s["cand"], s["target"])["iou"]
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                       extent=[HEAT_K[0], HEAT_K[-1], 0, len(HEAT_W) - 1])
        bi, bj = np.unravel_index(np.argmax(H), H.shape)
        ax.scatter([HEAT_K[bj]], [bi], marker="*", s=220, color="red",
                   edgecolor="w", label=f"best IoU {H[bi,bj]:.3f}")
        ax.set_yticks(range(len(HEAT_W)))
        ax.set_yticklabels([f"{w:.2g}" for w in HEAT_W], fontsize=7)
        ax.set_xlabel("threshold K"); ax.set_ylabel("TV weight")
        ax.set_title(f"run {run}  IoU(weight, K)   baseline {base[run]['iou']:.3f}")
        ax.legend(loc="lower right", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="IoU")
    fig.suptitle("TV variance mask -- IoU robustness over (weight, threshold)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, "tv_variance_heatmap.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # ---- save the best-weight mask per run -----------------------------------
    for run in RUNS:
        s = stash[run]
        best_w, best_iou, best_M = 0.0, -1, None
        for w in CURVE_W:
            M = tv_mask(s["z"], s["cand"], w)
            iou = score(M, s["target"])["iou"]
            if iou > best_iou:
                best_iou, best_w, best_M = iou, w, M
        np.save(os.path.join(OUT_DIR, f"tv_mask_run{run:04d}.npy"), best_M)
        print(f"run {run}: best TV IoU {best_iou:.3f} at weight={best_w:.3g}  "
              f"(close_open {base[run]['iou']:.3f})  -> saved")


if __name__ == "__main__":
    main()
