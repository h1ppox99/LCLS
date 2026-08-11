#!/usr/bin/env python3
"""
pixel_stats.py -- per-pixel statistics for the Jungfrau1M masking dataset.

Builds the mean and RMS (sqrt variance) maps that are the natural feature space
for detector masking, and plots the RMS map next to the mean image and the
reference masks.  Numpy-only (reads the frozen ./data arrays via dataset.py).

Statistics come from the *cleaned* run sums (per-event outliers already dropped):
    mean(x)     = Sum_dropped        / N
    mean(x^2)   = Sum_dropped_square / N
    var(x)      = mean(x^2) - mean(x)^2          (>= 0, verified)
    rms(x)      = sqrt(var)
where N is the number of shots the selection kept.

Why RMS for masking:
    * dead / disconnected pixels   -> rms ~ 0
    * hot / noisy / unstable pixels-> rms very large
    * good pixels                  -> rms in a narrow physical band
  This is the same signal the calibration `pixel_status` is built from, and it
  is far less sensitive to the per-ASIC additive offsets that dominate the raw
  sum image.

Outputs:
    outputs/features/{mean,rms}_run<NNNN>_{panel,asm}.npy
    images/rms_run<NNNN>_{gray,viridis}.png
Run:  python -m automask.studies.pixel_stats
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
from automask.dataset import load_mask
from automask.geometry import panel_to_asm
from automask.image_store import ImageStore
from automask.selection_presets import BEAM_ON_SELECTION

FEAT = os.path.join(AUTOMASK, "outputs", "features")
IMG_OUT = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FEAT, exist_ok=True)
os.makedirs(IMG_OUT, exist_ok=True)

RUNS = (389, 475)
OVERLAY = mcolors.ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.6)])


def mean_rms(run: int, form: str):
    """Per-pixel (mean, rms) over the selected shots, straight from ImageStore.

    The store already streams both moments in one pass, so the old frozen
    sum/sum-of-squares pair and the event count they had to be divided by are
    no longer needed.
    """
    store = ImageStore()
    mean = store.reduce(run, BEAM_ON_SELECTION, "mean", form).astype(np.float64)
    rms = store.reduce(run, BEAM_ON_SELECTION, "std", form).astype(np.float64)
    return mean, rms


def _n_shots(run: int) -> int:
    counts = ImageStore().counts(run, BEAM_ON_SELECTION, "mean") or {}
    return int(counts.get("n_used") or counts.get("n_selected") or 0)


def _status_mask(run: int) -> np.ndarray:
    """psana pixel status for `run`, in panel form (True == flagged bad).

    psana's convention is 1 == good; the project's is True == masked.
    """
    return ImageStore().calibration(run, "status_as_mask") == 0


def summarize(run: int):
    """Print per-pixel RMS statistics on the native panel geometry."""
    mean, rms = mean_rms(run, "panel")
    status = _status_mask(run)                                # True == flagged bad
    r = rms.ravel()
    finite = r[np.isfinite(r)]
    med = np.median(finite)
    # simple physical bands relative to the median good-pixel RMS
    dead = rms < 0.05 * med
    hot = rms > 20 * med
    print(f"\n--- run {run}  (N={_n_shots(run)} selected shots, "
          f"panel {rms.shape}) ---")
    print(f"  RMS   median {med:.3g}   p1 {np.percentile(finite,1):.3g}   "
          f"p99 {np.percentile(finite,99):.3g}   max {finite.max():.3g}")
    print(f"  candidate dead (<0.05*med): {dead.sum():6d} "
          f"({100*dead.mean():.3f}%)")
    print(f"  candidate hot  (>20*med)  : {hot.sum():6d} "
          f"({100*hot.mean():.3f}%)")
    cand = dead | hot
    inter = int((cand & status).sum())
    print(f"  psana pixel_status bad    : {int(status.sum()):6d} "
          f"({100*status.mean():.3f}%)")
    print(f"  of pixel_status-bad pixels, {100*inter/max(status.sum(),1):.1f}% "
          f"are extreme-RMS (dead|hot) -> RMS is a strong bad-pixel signal")


def save_features(run: int):
    for form in ("panel", "asm"):
        mean, rms = mean_rms(run, form)
        np.save(os.path.join(FEAT, f"mean_run{run:04d}_{form}.npy"),
                mean.astype(np.float32))
        np.save(os.path.join(FEAT, f"rms_run{run:04d}_{form}.npy"),
                rms.astype(np.float32))


def plot_rms(run: int, cmap: str):
    mean, rms = mean_rms(run, "asm")
    status = panel_to_asm(_status_mask(run), run)             # asm
    human = load_mask("human_Mask")

    # robust display ranges from the real (non-gap) pixels
    m = mean[mean != 0]
    mvmin, mvmax = np.percentile(m, 30), np.percentile(m, 99)
    rpos = rms[rms > 0]
    rnorm = mcolors.LogNorm(vmin=np.percentile(rpos, 5),
                            vmax=np.percentile(rpos, 99.5))

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    ax[0].imshow(mean, vmin=mvmin, vmax=mvmax, cmap=cmap)
    ax[0].set_title(f"run {run}  mean")
    im = ax[1].imshow(rms, norm=rnorm, cmap="magma")
    ax[1].set_title("RMS (sqrt variance, log)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[2].imshow(rms, norm=rnorm, cmap="magma")
    ax[2].imshow(status, cmap=OVERLAY)
    ax[2].set_title(f"RMS + pixel_status ({100*status.mean():.2f}%)")
    ax[3].imshow(rms, norm=rnorm, cmap="magma")
    ax[3].imshow(human, cmap=OVERLAY)
    ax[3].set_title(f"RMS + human_Mask ({100*human.mean():.2f}%)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"xppl1016922 run {run} — Jungfrau1M per-pixel statistics [{cmap}]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(IMG_OUT, f"rms_run{run:04d}_{cmap}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    for run in RUNS:
        summarize(run)
        save_features(run)
        for cmap in ("gray", "viridis"):
            plot_rms(run, cmap)
