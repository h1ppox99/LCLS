#!/usr/bin/env python3
"""
kw_sweep.py -- rigorous (K, W) hyperparameter sweep for the three TV-regularized
detectors in methods.py (variance, window_median, blackhat), on both runs.

For every (method, run) it renders ONE 4x4 image: 16 agreement maps over a grid
of 4 threshold values K (rows) x 4 TV weights W (columns). Each cell reuses the
exact production pipeline -- the statistic field is computed once, TV-denoised per
W, then thresholded per K with the method's own mode/padding -- so the picture is
what methods.py would actually produce at that (K, W).

Grids are centered toward LOWER K and HIGHER W than the current defaults. Each
cell title carries the combo IoU (score(floor | pick, human)) plus T-prec/T-rec
of the pick against the residual target; the current default (K, W) cell is
outlined blue and the best-combo-IoU cell green.

Outputs: outputs/figures/kw_sweep_<method>_run<NNNN>.png   (3 methods x 2 runs).
Run:  python kw_sweep.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
from dataset import score                                          # noqa: E402
from methods import (geometry_mask, fetch_calib_mask, load_all, load_umean,      # noqa: E402
                     variance_stat, window_median_stat, blackhat_stat,
                     tv_denoise, threshold_stat, pad_mask,
                     KVAR, WVAR, KWIN, WWIN, KBH, WBH)

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = (389, 475)

# Per-method sweep config. K = rows (top->bottom, low->high), W = cols
# (left->right, low->high). Grids sit at / below the default K and at / above the
# default W, per the "lower K, higher W" preference.
METHODS = {
    "variance": dict(mode="low", pad=False, default=(KVAR, WVAR),
                     K=[2.0, 2.5, 3.0, 3.5], W=[5.0, 10.0, 15.0, 20.0]),
    "window_median": dict(mode="low", pad=True, default=(KWIN, WWIN),
                           K=[1.5, 2.0, 2.5, 3.0], W=[5.0, 10.0, 15.0, 20.0]),
    "blackhat": dict(mode="high", pad=True, default=(KBH, WBH),
                      K=[1.5, 2.0, 2.5, 3.0], W=[5.0, 10.0, 15.0, 20.0]),
}


def agree_rgb(pred, truth):
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)      # TP green
    a[pred & ~truth] = (0.9, 0.0, 0.0)     # FP red
    a[~pred & truth] = (0.0, 0.3, 1.0)     # FN blue
    return a


def prep(run):
    """Floor, residual target, and each detector's continuous statistic field."""
    sumimg, _mean, ustd, human = load_all(run)
    umean = load_umean(run)
    real = sumimg != 0
    floor = geometry_mask(real) | fetch_calib_mask(run)
    target = human & ~floor
    fields = {"variance": variance_stat(ustd),
              "window_median": window_median_stat(umean, real),
              "blackhat": blackhat_stat(umean, real)}
    return dict(real=real, floor=floor, target=target, human=human, fields=fields)


def pick(field, k, w, mode, pad, real):
    """Reproduce the method_* pipeline for one (K, W): TV-denoise, threshold, pad."""
    M = threshold_stat(tv_denoise(field, w), k, mode) & real
    return (pad_mask(M) & real) if pad else M


def sweep_image(method, cfg, run, s):
    field = s["fields"][method]
    Ks, Ws = cfg["K"], cfg["W"]
    dK, dW = cfg["default"]

    # denoise once per W (independent of K), then threshold per K
    picks, ious = {}, np.zeros((len(Ks), len(Ws)))
    for j, w in enumerate(Ws):
        u = tv_denoise(field, w)
        for i, k in enumerate(Ks):
            M = threshold_stat(u, k, cfg["mode"]) & s["real"]
            if cfg["pad"]:
                M = pad_mask(M) & s["real"]
            picks[(i, j)] = M
            ious[i, j] = score(s["floor"] | M, s["human"])["iou"]
    bi, bj = np.unravel_index(int(np.argmax(ious)), ious.shape)

    fig, ax = plt.subplots(len(Ks), len(Ws), figsize=(18, 18))
    for i, k in enumerate(Ks):
        for j, w in enumerate(Ws):
            a = ax[i, j]
            M = picks[(i, j)]
            sc = score(M, s["target"])
            a.imshow(agree_rgb(M, s["target"])); a.set_xticks([]); a.set_yticks([])
            tags = []
            if np.isclose(k, dK) and np.isclose(w, dW):
                tags.append("default")
            if (i, j) == (bi, bj):
                tags.append("best")
            suffix = ("  [" + ", ".join(tags) + "]") if tags else ""
            color = ("green" if (i, j) == (bi, bj)
                     else "blue" if "default" in tags else "black")
            a.set_title(f"K={k:g}  W={w:g}{suffix}\nIoU={ious[i,j]:.3f}  "
                        f"Tp={sc['precision']:.2f}  Tr={sc['recall']:.2f}",
                        fontsize=10, color=color)
            if tags:  # outline default/best cells
                ec = "green" if (i, j) == (bi, bj) else "blue"
                a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes,
                                      fill=False, ec=ec, lw=3))

    fig.suptitle(f"xppl1016922 run {run} -- {method}: (K, W) sweep "
                 f"(rows K, cols W; green=TP red=FP blue=FN vs residual target; "
                 f"combo IoU in title, default={dK:g}/{dW:g})", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, f"kw_sweep_{method}_run{run:04d}.png")
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}   best combo IoU {ious[bi,bj]:.3f} at "
          f"K={Ks[bi]:g} W={Ws[bj]:g}")


def main():
    for run in RUNS:
        print(f"=== run {run} ===")
        s = prep(run)
        print(f"  floor IoU {score(s['floor'], s['human'])['iou']:.3f}  "
              f"residual target {int(s['target'].sum())} px")
        for method, cfg in METHODS.items():
            sweep_image(method, cfg, run, s)


if __name__ == "__main__":
    main()
