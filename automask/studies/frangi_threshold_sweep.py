#!/usr/bin/env python3
"""
studies/frangi_threshold_sweep.py -- visual sweep of the Frangi threshold k.

Fixes the filter at the locked production choice (sigmas=(1,2,3), beta=0.5,
gamma=None) and walks the detection threshold k across a geometric ladder
centred on the unsupervised auto-k (threshold_minimum). With `resp > k`,
LOWERING k masks MORE (more pixels clear a lower bar); raising it masks less.
The ladder leans toward the mask-more side so you can watch where the added
ridge pixels stop being genuine line/streak defects and start eating the
powder ring / background noise.

Each panel overlays, on the arcsinh sum image:
    blue  = the current mask (geometry+calib floor + production variance)
    red   = NEW pixels frangi adds at this k, beyond the current mask
title reports k (and its ratio to auto-k) and the added %.

One figure per run: frangi_threshold_sweep_run<NNNN>.png

Run:  python -m automask.studies.frangi_threshold_sweep
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automask.evaluation import EVAL_RUNS, reference_mask
from automask.sample import Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.masking import production_pipeline
from automask.dataset import score
from automask.regularization.frangi import frangi_ridges, auto_threshold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "outputs", "figures", "frangi_mask")
os.makedirs(OUTDIR, exist_ok=True)

# ratios of auto-k to visit: >1 masks LESS, <1 masks MORE (leaning mask-more).
# LADDERS[tag] -> the 6 ratios shown in that figure.
LADDERS = {
    "":     [2.0, 1.0, 0.5, 0.25, 0.12, 0.05],      # around auto-k
    "lowk": [0.05, 0.02, 0.008, 0.003, 0.001, 0.0003],  # deep into the floor
}


def current_mask(sample):
    pipe = production_pipeline("union")
    floor = pipe.floor(sample)
    var = next(d for d in pipe.evidence_channels if d.stat == "variance")
    return floor | var.pick(sample)


def sweep_run(run, ratios, tag):
    s = Sample.from_store(run, BEAM_ON_SELECTION, pipe.needs())
    cur = current_mask(s)
    domain = s.real & ~cur
    resp = frangi_ridges(s.mean)                 # locked defaults (1,2,3),0.5
    k0 = auto_threshold(resp, domain)

    base = np.arcsinh(s.mean / (np.nanmedian(np.abs(s.mean[s.real])) + 1e-9))
    vlo, vhi = np.nanpercentile(base[s.real], [2, 99])

    n = len(ratios); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 6.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    print(f"\nrun {run} [{tag or 'around'}]: auto-k = {k0:.4f}   "
          f"(current mask {100*cur.mean():.2f}%)")
    print(f"  {'ratio':>7s} {'k':>9s} {'added%':>8s} {'total%':>8s} {'IoU':>7s}")
    for ax, ratio in zip(axes, ratios):
        k = ratio * k0
        ridges = (resp > k) & domain
        full = cur | ridges
        iou = score(full, reference_mask(s.run))["iou"]
        print(f"  {ratio:7.4f} {k:9.5f} {100*ridges.mean():8.3f} "
              f"{100*full.mean():8.3f} {iou:7.3f}")

        ax.imshow(base.T, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
        blue = np.zeros((*s.mean.shape, 4)); blue[cur] = (0.1, 0.4, 1.0, 0.45)
        red = np.zeros((*s.mean.shape, 4)); red[ridges] = (1, 0, 0, 1)
        ax.imshow(np.transpose(blue, (1, 0, 2)), origin="lower")
        ax.imshow(np.transpose(red, (1, 0, 2)), origin="lower")
        lab = "auto-k" if ratio == 1.0 else f"{ratio:g}x auto-k"
        note = "  (mask MORE)" if ratio < 1 else ("  (mask less)" if ratio > 1 else "")
        ax.set_title(f"k={k:.5f}   {lab}{note}\n"
                     f"+{100*ridges.mean():.3f}% red   IoU {iou:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    span = "deep into the noise floor" if tag == "lowk" else "around auto-k"
    fig.suptitle(f"run {run}: Frangi threshold sweep, {span}  "
                 f"(s=(1,2,3), beta=0.5; blue=current mask, red=added ridges)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(OUTDIR, f"frangi_threshold_sweep{suffix}_run{run:04d}.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    return out


def main():
    outs = [sweep_run(run, LADDERS["lowk"], "lowk") for run in EVAL_RUNS]
    print("\nsaved:")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
