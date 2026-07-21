"""
viz.py -- shared masking figures (factored out of the one-off sweep studies).

  * agree_rgb           TP/FP/FN colour map (green/red/blue), the common overlay.
  * save_agreement      residual error map for one pipeline result on one run.
  * plot_results_heatmap 2-D IoU heatmap from a sweep results.csv over two knobs.
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def agree_rgb(pred, truth):
    """RGB agreement map: green=TP, red=FP, blue=FN, white=TN."""
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)
    a[pred & ~truth] = (0.9, 0.0, 0.0)
    a[~pred & truth] = (0.0, 0.3, 1.0)
    return a


def save_agreement(pred, floor, human, run, out, title=""):
    """Two-panel residual figure: pick-vs-residual-target error map + full mask."""
    from dataset import score
    target = human & ~floor
    resid = score(pred & ~floor, target)
    full = score(pred, human)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(agree_rgb(pred & ~floor, target)); ax[0].axis("off")
    ax[0].set_title(f"residual (green=TP red=FP blue=FN)\n"
                    f"T-IoU={resid['iou']:.3f} p={resid['precision']:.2f} "
                    f"r={resid['recall']:.2f}", fontsize=11)
    ax[1].imshow(agree_rgb(pred, human)); ax[1].axis("off")
    ax[1].set_title(f"full mask vs human\nIoU={full['iou']:.3f} "
                    f"p={full['precision']:.2f} r={full['recall']:.2f}", fontsize=11)
    fig.suptitle(title or f"run {run} — masking result", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    return out


def plot_results_heatmap(csv_path, xcol, ycol, out, metric="mean_iou"):
    """2-D `metric` heatmap over two swept columns from a sweep results.csv."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"empty results csv: {csv_path}")
    xs = sorted({float(r[xcol]) for r in rows})
    ys = sorted({float(r[ycol]) for r in rows})
    H = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        i = ys.index(float(r[ycol])); j = xs.index(float(r[xcol]))
        H[i, j] = float(r[metric])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(xs), 1.6 + 1.0 * len(ys)))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, len(xs), 0, len(ys)])
    bi, bj = np.unravel_index(np.nanargmax(H), H.shape)
    ax.scatter([bj + 0.5], [bi + 0.5], marker="*", s=220, color="red", edgecolor="w",
               label=f"best {metric}={H[bi,bj]:.3f}")
    ax.set_xticks(np.arange(len(xs)) + 0.5); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(np.arange(len(ys)) + 0.5); ax.set_yticklabels([f"{y:g}" for y in ys])
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{metric} over ({xcol}, {ycol})")
    fig.colorbar(im, ax=ax, fraction=0.046, label=metric)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out
