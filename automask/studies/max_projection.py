#!/usr/bin/env python3
"""
max_projection.py -- maximum-intensity projection (MIP) of the Jungfrau1M frames.

Study of a per-pixel *operation* for the masking project: instead of the run
SUM/mean (what the frozen dataset stores), take for every pixel its **maximum**
value over all beam-on frames of a run:

    mip[p] = max_over_events  calib_frame[event, p]        (beam on only)

Note the selection does NOT discriminate the CC/VCC beam branches, so the MIP
mixes both branch states; that is fine for a "was this pixel ever hit hard"
feature.

Outputs (under automask/outputs/):
    features/mip_run<NNNN>_{panel,asm}.npy      the MIP arrays (float32)
    figures/max_projection/mip_run<NNNN>.png    per-run figure (linear + log)
    figures/max_projection/mip_compare.png      both runs side by side
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from automask.geometry import panel_to_asm
from automask.shot_selection import Condition, ShotSelection

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(AUTOMASK, "outputs", "features")
FIG = os.path.join(AUTOMASK, "outputs", "figures", "max_projection")

RUNS = (389, 475)

#: Every beam-on frame, both branches, untrimmed and uncapped -- a max wants the
#: whole run, and clipping the intensity tails would defeat the point.
SELECTION = ShotSelection(where=(Condition(
    "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]", "==", 1,
),))


def compute_mip(run: int, selection: ShotSelection = SELECTION):
    """Per-pixel max over the selected calibrated frames. Returns (panel, stats)."""
    from automask.io.read_xtc import iter_calibrated
    from automask.utils import profile_run_values

    profile = profile_run_values(run, show=False)
    indices = selection.resolve(profile)
    print(f"[select] run {run:04d}: {profile.events} shots total, "
          f"{selection.describe(profile)['n_eligible']} eligible, "
          f"{indices.size} selected")

    mip = None
    n_used = 0
    for _, panel in iter_calibrated(run, indices, source=profile.source):
        mip = panel.copy() if mip is None else np.maximum(mip, panel)
        n_used += 1
        if n_used % 500 == 0:
            print(f"  run {run}: {n_used}/{indices.size} frames folded in",
                  flush=True)

    if mip is None:
        raise RuntimeError(f"run {run}: no calibrated frames read")
    print(f"[done] run {run}: MIP over {n_used} frames")
    return mip, {"n_selected": int(indices.size), "n_used": n_used}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _log_norm(asm: np.ndarray):
    """LogNorm over the real (non-gap, positive) assembled pixels."""
    pos = asm[asm > 0]
    return mcolors.LogNorm(vmin=max(np.percentile(pos, 50), 1.0),
                           vmax=np.percentile(pos, 99.9))


def plot_run(run: int, asm: np.ndarray, stats: dict):
    pos = asm[asm > 0]
    lin_vmax = np.percentile(pos, 99.5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))
    im0 = ax[0].imshow(asm, vmin=0, vmax=lin_vmax, cmap="viridis")
    ax[0].set_title(f"MIP (linear, clip @ p99.5={lin_vmax:.0f})")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(asm, norm=_log_norm(asm), cmap="magma")
    ax[1].set_title("MIP (log)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    for a in ax:
        a.axis("off")
    fig.suptitle(
        f"xppl1016922 run {run} — Jungfrau1M max-intensity projection "
        f"({stats['n_used']} beam-on frames)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, f"mip_run{run:04d}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def plot_compare(asms: dict, stats: dict):
    fig, ax = plt.subplots(1, len(asms), figsize=(6.5 * len(asms), 6.6))
    ax = np.atleast_1d(ax)
    for a, (run, asm) in zip(ax, asms.items()):
        im = a.imshow(asm, norm=_log_norm(asm), cmap="magma")
        a.set_title(f"run {run}  (MIP, log, "
                    f"{stats[run]['n_used']} frames)")
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("Jungfrau1M max-intensity projection — beam-on frames",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "mip_compare.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    os.makedirs(FEAT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    asms, stats = {}, {}
    for run in RUNS:
        panel, st = compute_mip(run)
        asm = panel_to_asm(panel, run).astype(np.float32)
        np.save(os.path.join(FEAT, f"mip_run{run:04d}_panel.npy"), panel)
        np.save(os.path.join(FEAT, f"mip_run{run:04d}_asm.npy"), asm)
        plot_run(run, asm, st)
        asms[run], stats[run] = asm, st
    plot_compare(asms, stats)


if __name__ == "__main__":
    main()
