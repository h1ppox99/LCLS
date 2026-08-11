#!/usr/bin/env python3
"""
studies/frangi_loghist.py -- visualize the Frangi-response distribution.

Reproduces the production pre-frangi mask (geometry+calib floor UNION the
production variance detector), runs Frangi on the sum image, and plots the
histogram of the response over the still-unmasked domain (real & ~current_mask)
in LOG space -- the claim being that this histogram is BIMODAL (a broad
noise-floor mode + a separated ridge mode) with a valley that
`threshold_minimum` locks onto. One figure per run + a linear-space companion so
the bimodality is visibly a log-space fact.

Run:  python -m automask.studies.frangi_loghist
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
from automask.regularization.frangi import frangi_ridges, auto_threshold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "outputs", "figures", "frangi_mask")
os.makedirs(OUTDIR, exist_ok=True)

FLOOR = 1e-9   # discard exact/near zeros (gaps, off-detector) before the log


def current_mask(sample):
    """geometry+calib floor UNION the production variance detector pick."""
    pipe = production_pipeline("union")
    floor = pipe.floor(sample)
    var = next(d for d in pipe.evidence_channels if d.stat == "variance")
    return floor | var.pick(sample)


def response(sample):
    """Frangi response on the sum image and the domain it is thresholded over."""
    resp = frangi_ridges(sample.mean)
    domain = sample.real & ~current_mask(sample)
    return resp, domain


def plot_run(run, ax_log, ax_lin):
    sample = Sample.from_store(run, BEAM_ON_SELECTION, pipe.needs())
    resp, domain = response(sample)
    v = resp[domain]
    v = v[v > FLOOR]
    k = auto_threshold(resp, domain)

    logv = np.log10(v)
    ax_log.hist(logv, bins=256, color="#3b6ea5", alpha=0.85)
    ax_log.axvline(np.log10(k), color="crimson", lw=2,
                   label=f"auto k = {k:.4g}  (log10 {np.log10(k):.2f})")
    ax_log.set_title(f"run {run}: Frangi response, LOG space (n={v.size})")
    ax_log.set_xlabel("log10(Frangi response)")
    ax_log.set_ylabel("pixels")
    ax_log.legend(loc="upper right", fontsize=9)

    # Linear space, same data -- to show the low mode swamps everything and why
    # Otsu/triangle/Li (which work in linear space) collapse to ~0.
    ax_lin.hist(v, bins=256, color="#8a8a8a", alpha=0.85)
    ax_lin.axvline(k, color="crimson", lw=2, label=f"auto k = {k:.4g}")
    ax_lin.set_yscale("log")
    ax_lin.set_title(f"run {run}: same response, LINEAR x (log y)")
    ax_lin.set_xlabel("Frangi response")
    ax_lin.set_ylabel("pixels (log)")
    ax_lin.legend(loc="upper right", fontsize=9)

    frac_above = float((v > k).mean())
    print(f"run {run}: n={v.size}  auto k={k:.5f}  "
          f"log10 k={np.log10(k):.3f}  above k: {100*frac_above:.3f}%  "
          f"response max={v.max():.4f}")


def main():
    runs = list(EVAL_RUNS)
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 4.2 * len(runs)))
    axes = np.atleast_2d(axes)
    for i, run in enumerate(runs):
        plot_run(run, axes[i, 0], axes[i, 1])
    fig.tight_layout()
    out = os.path.join(OUTDIR, "frangi_loghist.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
