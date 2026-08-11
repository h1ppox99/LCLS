#!/usr/bin/env python3
"""
studies/frangi_radialnorm.py -- does full per-bin radial normalization + polarity
let Frangi see the faint jet streaks?

Two questions:
  (1) Is a missed streak actually IN the Frangi response, or at the floor? -> we
      render the response FIELD directly (log), not just the thresholded mask.
  (2) Does preprocessing help? We compare three inputs to Frangi:
        raw    : the raw sum image (what the threshold-sweep study used)
        med    : current sigma_clipping field (azimuthal sigma-clip z-score)
        full   : per-bin FULL normalization (median sub + per-bin 1.4826*MAD)
      and both polarities (black_ridges False=bright ridge, True=dark ridge).

Filter: sigmas=(1,2,3,4,5) (extended to 5 px), beta=0.5, gamma=None.

Per run, two stacked figures:
  frangi_radialnorm_resp_run<NNNN>.png   log Frangi response fields
  frangi_radialnorm_mask_run<NNNN>.png   auto-thresholded added-mask overlays

Run:  python -m automask.studies.frangi_radialnorm
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
from automask.stats.sigma_clipping import sigma_clipping_stat
from automask.regularization.frangi import frangi_ridges, auto_threshold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "outputs", "figures", "frangi_mask")
os.makedirs(OUTDIR, exist_ok=True)

SIGMAS = (1.0, 2.0, 3.0, 4.0, 5.0)     # extended to 5 px
MIN_BIN = 30                            # min pixels for a trustworthy per-bin scale


def current_mask(sample):
    pipe = production_pipeline("union")
    floor = pipe.floor(sample)
    var = next(d for d in pipe.evidence_channels if d.stat == "variance")
    return floor | var.pick(sample)


def radial_full_norm(image, real, center, bin_width=3.0):
    """Per-annulus robust z: (x - median_bin) / (1.4826 * MAD_bin). Bins too small
    for a stable MAD fall back to the global robust scale. ~real set to 0."""
    image = image.astype(np.float64, copy=False)
    c0, c1 = center
    i0, i1 = np.indices(image.shape)
    bin_idx = (np.hypot(i0 - c0, i1 - c1) // bin_width).astype(np.int64)
    n_bins = int(bin_idx.max()) + 1
    med = np.zeros(n_bins)
    scale = np.zeros(n_bins)
    # global fallback scale
    allres = image[real] - np.median(image[real])
    gscale = 1.4826 * np.median(np.abs(allres)) or 1.0
    for b in range(n_bins):
        vals = image[(bin_idx == b) & real]
        if vals.size == 0:
            med[b], scale[b] = 0.0, gscale
            continue
        m = np.median(vals)
        s = 1.4826 * np.median(np.abs(vals - m))
        med[b] = m
        scale[b] = s if (vals.size >= MIN_BIN and s > 0) else gscale
    out = (image - med[bin_idx]) / scale[bin_idx]
    out[~real] = 0.0
    return out


def responses(sample):
    """dict label -> (input_field, frangi_response) for the input variants."""
    s = sample
    raw = s.mean.astype(np.float64)
    med = sigma_clipping_stat(s.mean, s.real, s.center)         # current
    full = radial_full_norm(s.mean, s.real, s.center)           # proposed
    out = {}
    out["raw (bright)"] = (raw, frangi_ridges(raw, sigmas=SIGMAS, black_ridges=False))
    out["med+globalz (bright)"] = (med, frangi_ridges(med, sigmas=SIGMAS, black_ridges=False))
    out["full norm (bright)"] = (full, frangi_ridges(full, sigmas=SIGMAS, black_ridges=False))
    out["full norm (dark)"] = (full, frangi_ridges(full, sigmas=SIGMAS, black_ridges=True))
    return out


def logshow(ax, resp, real, title):
    v = resp[real]
    v = v[v > 1e-12]
    vhi = np.percentile(v, 99.5) if v.size else 1.0
    disp = np.log10(np.clip(resp, 1e-9, None))
    m = np.ma.masked_where(~real, disp)
    im = ax.imshow(m.T, cmap="inferno", origin="lower",
                   vmin=np.log10(1e-4 * vhi + 1e-12), vmax=np.log10(vhi + 1e-12))
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def run_figs(run):
    s = Sample.from_store(run, BEAM_ON_SELECTION, pipe.needs())
    cur = current_mask(s)
    domain = s.real & ~cur
    resp = responses(s)

    base = np.arcsinh(s.mean / (np.nanmedian(np.abs(s.mean[s.real])) + 1e-9))
    vlo, vhi = np.nanpercentile(base[s.real], [2, 99])
    labels = list(resp.keys())

    # ---- figure A: response fields -------------------------------------------
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4.3 * (len(labels) + 1), 4.6))
    axes[0].imshow(base.T, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
    axes[0].set_title(f"run {run}: arcsinh(sum)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, lab in zip(axes[1:], labels):
        logshow(ax, resp[lab][1], s.real, f"Frangi resp: {lab}")
    fig.suptitle(f"run {run}: Frangi RESPONSE field (log) -- is the streak in it?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pa = os.path.join(OUTDIR, f"frangi_radialnorm_resp_run{run:04d}.png")
    fig.savefig(pa, dpi=125); plt.close(fig)

    # ---- figure B: auto-thresholded added-mask overlays ----------------------
    fig, axes = plt.subplots(1, len(labels), figsize=(5.6 * len(labels), 6.0))
    axes = np.atleast_1d(axes)
    print(f"\nrun {run}:  (current mask {100*cur.mean():.2f}%)")
    print(f"  {'input':24s} {'auto-k':>9s} {'added%':>8s} {'IoU':>7s}")
    for ax, lab in zip(axes, labels):
        r = resp[lab][1]
        k = auto_threshold(r, domain)
        added = (r > k) & domain if np.isfinite(k) else np.zeros_like(domain)
        iou = score(cur | added, reference_mask(s.run))["iou"]
        print(f"  {lab:24s} {k:9.4f} {100*added.mean():8.3f} {iou:7.3f}")
        ax.imshow(base.T, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
        blue = np.zeros((*s.mean.shape, 4)); blue[cur] = (0.1, 0.4, 1.0, 0.4)
        red = np.zeros((*s.mean.shape, 4)); red[added] = (1, 0, 0, 1)
        ax.imshow(np.transpose(blue, (1, 0, 2)), origin="lower")
        ax.imshow(np.transpose(red, (1, 0, 2)), origin="lower")
        ax.set_title(f"{lab}\nauto-k={k:.4f}  +{100*added.mean():.3f}%  IoU {iou:.3f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"run {run}: auto-thresholded ADDED mask (blue=current, red=added)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pb = os.path.join(OUTDIR, f"frangi_radialnorm_mask_run{run:04d}.png")
    fig.savefig(pb, dpi=125); plt.close(fig)
    return pa, pb


def main():
    outs = []
    for run in EVAL_RUNS:
        outs.extend(run_figs(run))
    print("\nsaved:")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
