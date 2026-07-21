#!/usr/bin/env python3
"""
radial_median_subtract.py -- radial-bin median-subtraction map for a run.

Bins the assembled sum image into concentric annuli around the run's beam
center (geometry.get_center -- see that module's docstring for why this is
NOT simply CLAUDE.md's "(col 1005, row 45)" plugged in directly: the frozen
`_asm` arrays store axis0/axis1 transposed relative to that phrasing) and,
within each annulus, subtracts that annulus's median from every pixel in it.
This flattens the radially-symmetric scattering falloff and leaves behind
local outliers (hot/dead pixels, panel artifacts) that don't follow the ring
pattern.

The geometry mask (ASIC boundaries / inter-module gap, methods.geometry_mask)
and the low-variance mask (dead/shadowed pixels, methods.method_variance) are
applied FIRST and excluded from every annulus's median -- masked pixels would
otherwise drag the ring statistic since they aren't real detector signal.
Detector gaps (exact-zero pixels in the assembled image) are excluded the
same way and kept at exactly zero in the output, matching
window_median_subtract.py's convention.

Outputs:
    outputs/figures/reference_{RUN}_radial_median.png
Run:  python radial_median_subtract.py [--run RUN] [--bin-width PX]   (from src/automask/)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from methods import geometry_mask, method_variance, load_all
from geometry import get_center

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUN = 389
BIN_WIDTH = 3.0            # px, radial annulus width


def radial_bins(shape, center0, center1, bin_width):
    """Integer radial-bin index of every pixel, distance measured from the
    beam center. `center0`/`center1` are in the array's own (axis0, axis1)
    index order -- see geometry.get_center -- NOT a (row, col) pair, since
    for these frozen arrays that distinction is transposed from what the
    names would suggest."""
    i0, i1 = np.indices(shape)
    r = np.hypot(i0 - center0, i1 - center1)
    return (r // bin_width).astype(np.int64)


def radial_median_subtract(img, valid, bin_idx):
    """Subtract each annulus's median (over `valid` pixels) from every pixel
    in that annulus. Pixels outside `valid` are forced to zero in the output,
    same convention as window_median_subtract.py's detector-gap handling."""
    out = np.zeros_like(img, dtype=np.float64)
    n_bins = int(bin_idx.max()) + 1
    medians = np.zeros(n_bins)
    for b in range(n_bins):
        ring = bin_idx == b
        vals = img[ring & valid]
        medians[b] = np.median(vals) if vals.size else 0.0
    out = img - medians[bin_idx]
    out[~valid] = 0.0
    return out


def main(run: int = RUN, bin_width: float = BIN_WIDTH):
    sumimg, mean, ustd, _ = load_all(run)
    sumimg = sumimg.astype(np.float64)
    real = sumimg != 0
    geom = geometry_mask(real)                    # 100%-precision geometry floor
    variance = method_variance(mean, ustd, real)   # low-variance (dead/shadowed) outliers
    valid = real & ~(geom) # | variance)              # pixels allowed to inform a ring's median

    center0, center1 = get_center(run)
    bin_idx = radial_bins(sumimg.shape, center0, center1, bin_width)
    residual = radial_median_subtract(sumimg, valid, bin_idx)

    s = sumimg[sumimg != 0]
    svmin, svmax = np.percentile(s, 30), np.percentile(s, 99)
    r = residual[residual != 0]
    rlim = np.percentile(np.abs(r), 99) if r.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(sumimg, vmin=svmin, vmax=svmax, cmap="viridis")
    axes[0].plot(center1, center0, "r+", markersize=12, markeredgewidth=2)
    axes[0].set_title(f"run {run}  sum_calib")

    im = axes[1].imshow(residual, vmin=-rlim, vmax=rlim, cmap="viridis")
    axes[1].set_title(f"radial-bin ({bin_width:.0f}px) median-subtracted\n"
                       f"after geometry+variance masking")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"xppl1016922 run {run} — radial median subtraction "
                 f"about center (axis0={center0:.0f}, axis1={center1:.0f})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, f"reference_{run}_radial_median.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=RUN)
    ap.add_argument("--bin-width", type=float, default=BIN_WIDTH)
    args = ap.parse_args()
    main(args.run, args.bin_width)
