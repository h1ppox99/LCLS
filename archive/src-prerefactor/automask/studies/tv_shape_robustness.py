#!/usr/bin/env python3
"""
tv_shape_robustness.py -- how TV weight trades region-fill against SHAPE robustness.

The variance IoU-vs-weight plateau on runs 475/389 is flat from weight 4..16 only
because their sole residual defect is the fat beam-stop BLOB -- a low-perimeter
shape TV barely touches. To see what a large weight would do to OTHER defect shapes
(the whole point of a run-agnostic masker), we inject synthetic defects of known
shape into the real variance z-score of run 475 and watch which survive TV before
the K=3.5 threshold:

  isolated 1px | 1px lines (V/H/diagonal) | 2px line | 15x15 blob (control)

TV penalizes boundary length, so high-perimeter/low-area shapes (thin lines, points)
are erased first; the blob survives longest. The figure shows the recovered mask in
the injection window at increasing weight -- thin shapes vanish by weight~2-4, the
blob itself starts eroding past ~8. Conclusion: weight~4 is the elbow (consolidates
salt-and-pepper, still preserves thin structure and full blob coverage); weight>=8
is region-only and needs K re-tuned as the blob erodes.

Run:  python tv_shape_robustness.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
import methods as M                                    # noqa: E402

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RUN, K, Z0 = 475, 3.5, -6.0                            # inject clear (< -K) low-var defects
WEIGHTS = [0, 2, 4, 8, 16]
HALF = (80, 110)                                       # injection window half-size (rows, cols)


def variance_z(ustd):
    m = ustd > 0
    logr = np.log10(ustd[m]); med = np.median(logr)
    mad = np.median(np.abs(logr - med)) * 1.4826 + 1e-12
    z = np.zeros(ustd.shape); z[m] = (logr - med) / mad
    return z


def quiet_anchor(z, hr, hc, thr=K):
    """Top-left of the (2hr x 2hc) window with the FEWEST pre-existing |z|>thr hits,
    so injected shapes are read against clean background."""
    from scipy import ndimage as ndi
    hot = (np.abs(z) > thr).astype(float)
    dens = ndi.uniform_filter(hot, size=(2 * hr + 1, 2 * hc + 1))
    dens[: hr] = dens[-hr:] = 1e9  # keep window in-bounds
    dens[:, : hc] = dens[:, -hc:] = 1e9
    r, c = np.unravel_index(np.argmin(dens), dens.shape)
    return r - hr, c - hc


def main():
    sumimg, mean, ustd, human = M.load_all(RUN)
    z = variance_z(ustd)
    hr, hc = HALF
    r0, c0 = quiet_anchor(z, hr, hc)
    H, W = 2 * hr, 2 * hc

    zc = z.copy()
    shapes = {}
    def put(name, coords):
        for (rr, cc) in coords:
            zc[rr, cc] = Z0
        shapes[name] = coords
    put("isolated 1px", [(r0 + 20, c0 + 20)])
    put("1px V-line",   [(r0 + 40 + i, c0 + 45) for i in range(40)])
    put("1px H-line",   [(r0 + 130, c0 + 60 + i) for i in range(40)])
    put("2px V-line",   [(r0 + 40 + i, c0 + 100 + j) for i in range(40) for j in range(2)])
    put("1px diagonal", [(r0 + 40 + i, c0 + 140 + i) for i in range(40)])
    put("15x15 blob",   [(r0 + 130 + i, c0 + 150 + j) for i in range(15) for j in range(15)])

    truth = np.zeros((H, W), bool)
    for coords in shapes.values():
        for (rr, cc) in coords:
            truth[rr - r0, cc - c0] = True

    bw = mcolors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    axes = ax.ravel()
    axes[0].imshow(truth, cmap=bw)
    axes[0].set_title("injected defects (ground truth)\n"
                      "isolated / 1px lines / 2px line / blob", fontsize=11)
    axes[0].axis("off")

    for a, w in zip(axes[1:], WEIGHTS):
        u = zc if w == 0 else denoise_tv_chambolle(zc, weight=w)
        rec = (u < -K)[r0:r0 + H, c0:c0 + W]
        img = np.ones((H, W, 3))
        img[rec & truth] = (0.0, 0.7, 0.0)     # survived (TP)
        img[~rec & truth] = (0.0, 0.3, 1.0)    # erased  (FN)
        img[rec & ~truth] = (0.9, 0.0, 0.0)    # spurious (FP)
        surv = (rec & truth).sum() / truth.sum()
        a.imshow(img)
        a.set_title(f"TV weight={w}  --  {100*surv:.0f}% of defect survives\n"
                    f"green=survives blue=erased red=spurious", fontsize=11)
        a.axis("off")

    fig.suptitle(f"xppl1016922 run {RUN} -- TV shape robustness: thin defects are "
                 f"erased first, blob last (threshold K={K})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, "tv_shape_robustness.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
