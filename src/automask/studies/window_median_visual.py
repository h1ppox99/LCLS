#!/usr/bin/env python3
"""
window_median_visual.py -- visual inspection of methods.method_window_median,
the tile-median-subtraction detector that IS scored but NOT (yet) in the
pipeline (see its docstring in methods.py: a real, run-asymmetric win --
389: 0.691 -> 0.797 IoU, 475: 0.766 -> 0.763).

IMPORTANT: this must run on `umean` (load_umean()), the genuine lit-beam
mean -- NOT load_all()'s `mean`, which is a beam-OFF dark/pedestal frame with
no scattering structure. An earlier version of this script fed it `mean` by
mistake; the mask still "looked plausible" and scored non-trivially, which is
exactly why that bug is dangerous -- it doesn't crash or produce visibly
empty output, see automask-dark-frame-vs-lit-mean in project memory.

The IoU/precision/recall numbers alone don't show WHERE the mask hits or
misses, and human_Mask itself is a hand-drawn, imperfect ground truth (see
CLAUDE.md / project memory), so a metric delta of a few px in a hundred
thousand can be ground-truth noise rather than a real difference. This script
renders the mask directly against the umean image and against human_Mask so
that can be judged by eye instead of by IoU alone.

Per run: umean image with the window_median footprint overlaid (outside the
geometry+calib floor), the same footprint coloured TP(green)/FP(red) against
the residual target, and a zoom on its densest cluster.

Run:  python window_median_visual.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
import methods as M                                    # noqa: E402
from dataset import score                              # noqa: E402

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RUNS = (475, 389)
OVERLAY = mcolors.ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.6)])  # red where masked


def footprint_rgb(pred, target):
    """White bg; green = pred hits target (TP), red = pred misses (FP)."""
    img = np.ones((*target.shape, 3))
    img[pred & target] = (0.0, 0.7, 0.0)
    img[pred & ~target] = (0.9, 0.0, 0.0)
    return img


def densest_window(mask, half=90):
    """Crop centred on the densest cluster of `mask` (smoothed-footprint argmax)."""
    dens = ndi.uniform_filter(mask.astype(float), size=2 * half + 1)
    r, c = np.unravel_index(np.argmax(dens), dens.shape)
    H, W = mask.shape
    r0, r1 = max(0, r - half), min(H, r + half)
    c0, c1 = max(0, c - half), min(W, c + half)
    return r0, r1, c0, c1


def run_row(axes, run):
    sumimg, mean, ustd, human = M.load_all(run)
    umean = M.load_umean(run)              # genuine lit-beam mean -- `mean` is a dark frame
    real = sumimg != 0
    floor = M.geometry_mask(real) | M.fetch_calib_mask(run)
    target = human & ~floor
    wmed = M.method_window_median(umean, real)
    wmed_o = wmed & ~floor

    st = score(wmed_o, target)
    combo_base = score(floor | M.method_variance(mean, ustd, real) | M.method_blackhat(mean, real), human)
    combo_with = score(floor | M.method_variance(mean, ustd, real) | M.method_blackhat(mean, real) | wmed, human)

    s = umean[real]
    vmin, vmax = np.percentile(s, 30), np.percentile(s, 99)

    axes[0].imshow(umean, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].imshow(wmed_o, cmap=OVERLAY)
    axes[0].set_title(f"run {run}: umean (lit) + window_median footprint\n"
                       f"({100*wmed_o.mean():.2f}% of chip, outside floor)", fontsize=10)

    axes[1].imshow(footprint_rgb(wmed_o, target))
    axes[1].set_title(f"vs residual target: green=TP red=FP\n"
                       f"T-prec={st['precision']:.3f} T-rec={st['recall']:.3f}", fontsize=10)

    crop = densest_window(wmed_o)
    r0, r1, c0, c1 = crop
    axes[2].imshow(footprint_rgb(wmed_o, target)[r0:r1, c0:c1])
    axes[2].set_title(f"zoom on densest cluster\n{int(wmed_o.sum())} px added", fontsize=10)

    axes[3].imshow(M._agree(floor | wmed, human))
    axes[3].set_title(f"floor|window_median vs human_Mask\n"
                       f"green=TP red=FP blue=FN", fontsize=10)

    axes[4].axis("off")
    axes[4].text(0.0, 0.5,
                 f"combo (geom|calib|variance|blackhat)\n"
                 f"  IoU {combo_base['iou']:.4f}  prec {combo_base['precision']:.3f}  "
                 f"rec {combo_base['recall']:.3f}\n\n"
                 f"+ window_median\n"
                 f"  IoU {combo_with['iou']:.4f}  prec {combo_with['precision']:.3f}  "
                 f"rec {combo_with['recall']:.3f}\n\n"
                 f"(ground truth is a hand-drawn mask,\n"
                 f"not exact -- read px-level agreement,\n"
                 f"not just the IoU delta)",
                 fontsize=10, va="center", transform=axes[4].transAxes)

    for a in axes:
        a.axis("off")


def main():
    fig, axes = plt.subplots(len(RUNS), 5, figsize=(24, 5 * len(RUNS)))
    if len(RUNS) == 1:
        axes = axes[None, :]
    for row, run in zip(axes, RUNS):
        run_row(row, run)
    fig.suptitle("xppl1016922 -- method_window_median: where it hits/misses "
                 "(NOT in the production combo, see methods.py docstring)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, "window_median_visual.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
