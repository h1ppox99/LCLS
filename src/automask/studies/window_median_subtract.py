#!/usr/bin/env python3
"""
window_median_subtract.py -- local (tile) median-subtraction map for run 389,
swept over window size.

Tiles the assembled sum image into non-overlapping win x win-pixel windows and,
within each window, subtracts that window's median from every pixel in it.
This flattens the slowly-varying illumination / per-ASIC offset structure and
leaves behind local outliers (hot/dead pixels, panel-edge artifacts) that a
single global mean would wash out. Detector gaps (exact-zero pixels in the
assembled image) are excluded from each window's median and kept at exactly
zero in the output.

The geometry mask (ASIC boundaries / inter-module gap, methods.geometry_mask)
and the low-variance mask (dead/shadowed pixels, methods.method_variance) are
applied to the sum image FIRST -- masked pixels are zeroed out exactly like the
existing detector gaps, so they don't pollute any window's median or show up
as spurious local outliers.

Repeats the subtraction for every window size in WINS and lays all of them out
in one grid figure next to the reference sum image, so the effect of window
size is directly comparable.

Outputs:
    outputs/figures/window_median_run0389_viridis_sweep.png
Run:  python window_median_subtract.py     (from src/automask/)
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import load_image
from methods import geometry_mask, method_variance, load_all

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUN = 389
WINS = (15, 18, 21, 24, 27, 30)
OVERLAY = mcolors.ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.55)])  # red where masked


def window_median_subtract(img: np.ndarray, win: int) -> np.ndarray:
    """Subtract each win x win tile's median (over nonzero pixels) from itself.

    Gap pixels (exact zero, i.e. outside the physical panels) are excluded
    from the median and forced back to zero in the output, so they don't
    read as artificial outliers.
    """
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(0, H, win):
        i1 = min(i + win, H)
        for j in range(0, W, win):
            j1 = min(j + win, W)
            tile = img[i:i1, j:j1]
            valid = tile[tile != 0]
            med = np.median(valid) if valid.size else 0.0
            residual = tile - med
            residual[tile == 0] = 0.0
            out[i:i1, j:j1] = residual
    return out


def main():
    sumimg = load_image(f"sum_calib_run{RUN:04d}", "asm").astype(np.float64)
    _, mean, ustd, _ = load_all(RUN)                  # mean/ustd features for method_variance
    real = sumimg != 0

    geom = geometry_mask(real)                         # 100%-precision geometry floor
    variance = method_variance(mean, ustd, real)        # low-variance (dead/shadowed) outliers
    applied = (geom | variance) & real

    masked_sum = sumimg.copy()
    masked_sum[applied] = 0.0                          # treat masked px like detector gaps

    s = sumimg[sumimg != 0]
    svmin, svmax = np.percentile(s, 30), np.percentile(s, 99)

    n_panels = 1 + len(WINS)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows))
    axes = axes.ravel()

    axes[0].imshow(sumimg, vmin=svmin, vmax=svmax, cmap="viridis")
    axes[0].imshow(applied, cmap=OVERLAY)
    axes[0].set_title(f"run {RUN}  sum image + geometry|variance mask "
                       f"({100*applied.mean():.1f}%)")

    for ax, win in zip(axes[1:], WINS):
        residual = window_median_subtract(masked_sum, win)
        r = residual[residual != 0]
        rlim = np.percentile(np.abs(r), 99)
        im = ax.imshow(residual, vmin=-rlim, vmax=rlim, cmap="viridis")
        ax.set_title(f"{win}x{win} window median-subtracted")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"xppl1016922 run {RUN} — local median subtraction vs window size "
                 f"after geometry+variance masking [viridis]", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, f"window_median_run{RUN:04d}_viridis_sweep.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
