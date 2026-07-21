#!/usr/bin/env python3
"""
max_projection.py -- maximum-intensity projection (MIP) of the Jungfrau1M frames.

Study of a per-pixel *operation* for the masking project: instead of the run
SUM/mean (what the frozen dataset stores), take for every pixel its **maximum**
value over all x-ray-on frames of a run:

    mip[p] = max_over_events  calib_frame[event, p]        (x-ray on only)

Why look at it: the sum/mean is dominated by the steady diffuse/powder signal,
so a dead pixel and a merely-dim pixel look similar and Bragg flashes average
away.  The max instead lights up every pixel that was *ever* hit hard -- Bragg
spots, hot/sparking pixels, and single-event cosmics -- while dead/disconnected
pixels stay flat.  It is a complementary feature to mean/RMS for masking.

This needs the raw per-event frames, which live only in the XTC (the small-data
file keeps just run Sums), so this study runs under psana:

    source psana_env.sh && python -m automask.studies.max_projection

x-ray-on selection is done *in-stream* from the BMMON i0 monitor
(XPP-SB2-BMMON.TotalIntensity), thresholded at I0_MIN.  x-ray-off (dropped)
shots have i0 ~ 0 while on-shots are 1e3-1e5, so the cut is unambiguous.  Doing
it from i0 -- rather than aligning against small-data lightStatus/xray by event
index -- is what lets run 389 work: locally it is a single, truncated stream
(s01, ~43%), whose events are a subset that would not line up with the merged
small-data index.

Outputs (under automask/outputs/):
    features/mip_run<NNNN>_{panel,asm}.npy      the MIP arrays (float32)
    figures/max_projection/mip_run<NNNN>.png    per-run figure (linear + log)
    figures/max_projection/mip_compare.png      both runs side by side
"""
from __future__ import annotations
import os
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))               # .../automask/studies
AUTOMASK = os.path.dirname(HERE)                               # .../automask
ROOT = os.path.dirname(AUTOMASK)                              # .../LCLS
XTC_DIR = os.path.join(ROOT, "xtc")
SMALLDATA = os.path.join(ROOT, "hdf5", "smalldata")

FEAT = os.path.join(AUTOMASK, "outputs", "features")
FIG = os.path.join(AUTOMASK, "outputs", "figures", "max_projection")

DET = "jungfrau1M_alcove"
ASM_SHAPE = (1064, 1030)
RUNS = (389, 475)
I0_MIN = 100.0            # BMMON TotalIntensity threshold for "x-ray on"


# ---------------------------------------------------------------------------
# geometry: reuse the small-data panel -> assembled-image map (same as
# producers/extract_dataset.assemble), so the MIP lands in the (1064,1030)
# frame the rest of the project works in.
# ---------------------------------------------------------------------------
def load_ixiy(run: int):
    with h5py.File(os.path.join(SMALLDATA, f"xppl1016922_Run{run:04d}.h5"), "r") as f:
        g = f[f"UserDataCfg/{DET}"]
        return g["ix"][()].astype(np.int64), g["iy"][()].astype(np.int64)


def assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    out = np.zeros(ASM_SHAPE, dtype=panel.dtype)
    out[ix, iy] = panel
    return out


def open_run(run: int):
    """(DataSource, files). Explicit-file open -- works for the multi-stream run
    475 and the single truncated stream of run 389 alike (see CLAUDE.md)."""
    import psana
    calib_dir = os.path.join(
        os.environ.get("SIT_PSDM_DATA", os.path.join(os.path.dirname(ROOT), "psdm")),
        "xpp", "xppl1016922", "calib")
    if os.path.isdir(calib_dir):
        psana.setOption("psana.calib-dir", calib_dir)
    files = sorted(glob.glob(
        os.path.join(XTC_DIR, f"xppl1016922-r{run:04d}-s0*-c00.xtc")))
    if not files:
        raise FileNotFoundError(f"no XTC streams for run {run} in {XTC_DIR}")
    print(f"[psana] run {run}: opening {len(files)} stream(s)")
    return psana.DataSource(*files), files


# ---------------------------------------------------------------------------
# the MIP itself
# ---------------------------------------------------------------------------
def compute_mip(run: int):
    """Per-pixel max over x-ray-on calibrated frames. Returns (mip_panel, stats)."""
    import psana
    ds, _ = open_run(run)
    det = psana.Detector(DET)
    try:
        i0det = psana.Detector("XPP-SB2-BMMON")               # ipm2
    except Exception:
        i0det = None
        print("[warn] BMMON unavailable -> keeping every decoded frame")

    mip = None                       # running per-pixel maximum (panel geometry)
    n_seen = n_on = n_used = 0
    for evt in ds.events():
        n_seen += 1
        # x-ray-on gate from i0 (skip dropped shots)
        if i0det is not None:
            d = i0det.get(evt)
            i0 = float(d.TotalIntensity()) if d is not None else np.nan
            if not (i0 > I0_MIN):
                continue
        n_on += 1
        cal = det.calib(evt)                                  # (2,512,1024) or None
        if cal is None:
            continue
        cal = cal.astype(np.float32)
        mip = cal.copy() if mip is None else np.maximum(mip, cal)
        n_used += 1
        if n_used % 500 == 0:
            print(f"  run {run}: {n_used} frames folded into MIP (scanned {n_seen})")

    if mip is None:
        raise RuntimeError(f"run {run}: no calibrated x-ray-on frames read")
    stats = dict(n_seen=n_seen, n_on=n_on, n_used=n_used)
    print(f"[done] run {run}: MIP over {n_used} x-ray-on frames "
          f"(scanned {n_seen}, i0>{I0_MIN:g} passed {n_on})")
    return mip, stats


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
        f"({stats['n_used']} x-ray-on frames)", fontsize=13)
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
    fig.suptitle("Jungfrau1M max-intensity projection — x-ray-on frames",
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
        mip_panel, st = compute_mip(run)
        ix, iy = load_ixiy(run)
        asm = assemble(mip_panel, ix, iy)
        np.save(os.path.join(FEAT, f"mip_run{run:04d}_panel.npy"), mip_panel)
        np.save(os.path.join(FEAT, f"mip_run{run:04d}_asm.npy"),
                asm.astype(np.float32))
        plot_run(run, asm, st)
        asms[run], stats[run] = asm, st
    plot_compare(asms, stats)


if __name__ == "__main__":
    main()
