#!/usr/bin/env python3
"""
tv_blackhat_visual.py -- visual proof that TV is the WRONG post-processing for the
black-hat channel (it erases the sparse dark specks that padding is meant to grow).

For each run we take the black-hat response R (dark-feature morphology), standardize
it to a robust-MAD z-score, and compare post-processings of the SAME response:

  raw   (u>k, no growth)         -- what TV reduces to at weight=0
  pad   (dilate by PAD)          -- the CURRENT production choice
  TV w  (denoise_tv_chambolle)   -- weight = 1, 4, 8

Each panel draws the black-hat mask footprint OUTSIDE the geometry+calib floor,
coloured green where it hits the residual target (human & ~floor) and red where it
misses (false positive).  The bottom row zooms on the black-hat activity so the
speck-by-speck erosion under TV is visible.  Titles carry T-prec / T-rec of the
channel and the resulting combo IoU (floor | variance | blackhat_variant vs human).

Run:  python tv_blackhat_visual.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from skimage.morphology import black_tophat, disk
from skimage.restoration import denoise_tv_chambolle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
import methods as M                                    # noqa: E402
from dataset import score                              # noqa: E402

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RUNS = (475, 389)
RADIUS, K = 5, 6.0        # black-hat scale + threshold, as in methods.method_blackhat


def bhat_z(mean, real, radius=RADIUS):
    """Robust-MAD z-score of the black-hat (dark-feature) response, 0 off-chip."""
    filled = np.where(real, mean, np.median(mean[real]))
    R = black_tophat(filled, footprint=disk(radius))
    med = np.median(R[real]); mad = np.median(np.abs(R[real] - med)) * 1.4826 + 1e-12
    z = np.zeros_like(R); z[real] = (R[real] - med) / mad
    return z


def footprint_rgb(pred, target, crop=None):
    """White bg; green = pred hits target (TP), red = pred misses (FP)."""
    img = np.ones((*target.shape, 3))
    img[pred & target] = (0.0, 0.7, 0.0)
    img[pred & ~target] = (0.9, 0.0, 0.0)
    if crop is not None:
        r0, r1, c0, c1 = crop
        img = img[r0:r1, c0:c1]
    return img


def densest_window(mask, half=90):
    """Crop (2*half square) centred on the densest cluster of `mask`, found by
    smoothing the footprint and taking the argmax -- so the zoom lands where the
    black-hat specks actually pile up, not on the full scattered bounding box."""
    from scipy import ndimage as ndi
    dens = ndi.uniform_filter(mask.astype(float), size=2 * half + 1)
    r, c = np.unravel_index(np.argmax(dens), dens.shape)
    H, W = mask.shape
    r0, r1 = max(0, r - half), min(H, r + half)
    c0, c1 = max(0, c - half), min(W, c + half)
    return r0, r1, c0, c1


def run_fig(run):
    sumimg, mean, ustd, human = M.load_all(run)
    real = sumimg != 0
    floor = M.geometry_mask(real) | M.fetch_calib_mask(run)
    target = human & ~floor
    var = M.method_variance(mean, ustd, real)
    base = floor | var                       # everything the blackhat is added on top of
    z = bhat_z(mean, real)

    def variant(kind, w=None):
        if kind == "pad":
            return M.method_blackhat(mean, real, radius=RADIUS, k=K)
        u = z if (w in (0, None)) else denoise_tv_chambolle(z, weight=w)
        return (u > K) & real

    masks = [("raw (no growth)", variant("raw", 0)),
             ("pad  [CURRENT]", variant("pad")),
             ("TV weight=1", variant("tv", 1)),
             ("TV weight=4", variant("tv", 4)),
             ("TV weight=8", variant("tv", 8))]

    crop = densest_window(variant("pad") & ~floor)   # zoom on the densest black-hat cluster

    fig, ax = plt.subplots(2, 6, figsize=(24, 8))
    # continuous response, full + cropped, as reference in column 0
    for a, (r, title) in zip(ax[:, 0], [(np.clip(z, -2, 12), "black-hat z-response"),
                                        (None, "same, zoom")]):
        if r is None:
            zc = np.clip(z, -2, 12)
            zc = zc[crop[0]:crop[1], crop[2]:crop[3]] if crop else zc
            im = a.imshow(zc, cmap="inferno")
        else:
            im = a.imshow(r, cmap="inferno")
        a.set_title(title, fontsize=10); a.axis("off")

    for col, (name, mk) in enumerate(masks, start=1):
        mo = mk & ~floor
        st = score(mo, target); sc = score(base | mk, human)
        ax[0, col].imshow(footprint_rgb(mk, target))
        ax[0, col].set_title(f"{name}\nT-prec={st['precision']:.2f} "
                             f"T-rec={st['recall']:.3f}\ncombo IoU={sc['iou']:.4f}",
                             fontsize=10); ax[0, col].axis("off")
        ax[1, col].imshow(footprint_rgb(mk, target, crop))
        ax[1, col].set_title(f"{name} (zoom)\n{int(mo.sum())} px added", fontsize=10)
        ax[1, col].axis("off")

    fig.suptitle(f"xppl1016922 run {run} -- black-hat post-processing: pad grows the "
                 f"sparse specks (green=TP red=FP), TV erases them (T-rec -> 0)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, f"tv_blackhat_visual_run{run:04d}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    for r in RUNS:
        run_fig(r)
