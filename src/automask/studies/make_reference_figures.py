#!/usr/bin/env python3
"""
make_reference_figures.py -- visual reference PNGs for the masking dataset.

For each calibrated sum image, draw the sum next to every reference mask
overlaid on it, so you can eyeball what each mask covers.  Output -> ./images/.
Numpy-only + matplotlib (no psana).
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
from dataset import load_image, load_mask

OUT = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

OVERLAY = mcolors.ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.55)])  # red where masked


def robust_range(img):
    """Per-image display range from the non-zero, finite pixels (full-run sums
    have very different scales per run, so fixed vmin/vmax saturate to white)."""
    v = img[(img != 0) & np.isfinite(img)]
    return np.percentile(v, 30), np.percentile(v, 99)


def panel(ax, img, mask=None, title="", cmap="gray"):
    vmin, vmax = robust_range(img)
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    if mask is not None:
        ax.imshow(mask, cmap=OVERLAY)
        title += f"  ({100*mask.mean():.2f}% masked)"
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def figure_for_run(run: int, cmap: str):
    img = load_image(f"sum_calib_run{run:04d}")
    refs = [("human_Mask", load_mask("human_Mask")),
            ("cmask", load_mask(f"cmask_run{run:04d}")),
            ("mask", load_mask(f"mask_run{run:04d}")),
            ("statusMask", load_mask(f"statusMask_run{run:04d}"))]

    fig, axes = plt.subplots(1, len(refs) + 1, figsize=(4 * (len(refs) + 1), 5))
    panel(axes[0], img, None, f"run {run}  sum_calib", cmap=cmap)
    for ax, (name, m) in zip(axes[1:], refs):
        panel(ax, img, m, name, cmap=cmap)
    fig.suptitle(f"xppl1016922 run {run} — Jungfrau1M sum + reference masks "
                 f"[{cmap}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"reference_run{run:04d}_{cmap}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    for run in (389, 475):
        for cmap in ("gray", "viridis"):
            figure_for_run(run, cmap)
