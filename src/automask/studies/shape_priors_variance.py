#!/usr/bin/env python3
"""
shape_priors_variance.py -- stage-0 for the "fit shapes instead of morphology" study.

Goal of the wider investigation: replace the pad_mask / close_open post-processing
of methods.py with a principled regularizer that trades OFF between

    (a) fidelity to the raw variance-outlier detection, and
    (b) shape priors  (isolated islands / lines / rectangles / curves),

under a single control parameter.  Morphological open/close is a fixed,
non-tunable version of that trade-off; we want the tunable, robust version.

This script produces the INPUT that every candidate regularizer consumes, for
BOTH run 475 and run 389:

  G       geometry floor            (methods.geometry_mask, run-agnostic)
  z       continuous low-variance score field  z(p) = -MAD_z(log10 uSTD)
          (high z == "wants to be masked"); the field a TV / graph-cut / MRF
          regularizer would denoise.  Defined only on real & ~G.
  raw     RAW binary outliers  (z > k) & real & ~G     -- salt & pepper, the
          point cloud a RANSAC / Hough primitive fit would consume.
  clean   the CURRENT post-processing  close_open(raw)  -- the baseline to beat.
  target  human & ~G                 -- the residual bad pixels we must recover.

Saves the arrays to outputs/masks/shape_prior_stage0/ and a 2x4 comparison
figure per run to outputs/figures/.  Nothing here fits a shape yet -- it just
freezes the substrate so the shape-fitting experiments are apples-to-apples.

Run:  python shape_priors_variance.py     (from src/automask/studies/)
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
from dataset import load_image, score                      # noqa: E402
from methods import geometry_mask, close_open, load_all    # noqa: E402

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
OUT_DIR = os.path.join(AUTOMASK, "outputs", "masks", "shape_prior_stage0")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = (475, 389)
K = 7.0          # same robust-MAD cut methods.method_variance uses


def variance_score(ustd):
    """Continuous low-variance score z: high == strongly wants masking.
    Robust-MAD z-score of log10(uSTD), negated so LOW variance -> HIGH score.
    Undefined pixels (uSTD==0) get -inf so they never score as outliers."""
    m = ustd > 0
    z = np.full(ustd.shape, -np.inf)
    logr = np.log10(ustd[m])
    med = np.median(logr)
    mad = np.median(np.abs(logr - med)) * 1.4826 + 1e-12
    z[m] = -(logr - med) / mad          # negate: low variance -> positive score
    return z


def stage0(run):
    sumimg, mean, ustd, human = load_all(run)
    real = sumimg != 0
    G = geometry_mask(real)                 # 100%-precision floor
    cand = real & ~G                        # where the detectors are allowed to act
    target = human & ~G                     # residual bad pixels to recover

    z = variance_score(ustd)
    z_field = np.where(cand, z, np.nan)     # score field on the candidate set only
    raw = (z > K) & cand                    # RAW salt-and-pepper outliers
    clean = close_open(raw) & real          # CURRENT baseline post-processing
    return dict(sumimg=sumimg, real=real, G=G, cand=cand, target=target,
                z=z, z_field=z_field, raw=raw, clean=clean, human=human)


def report(run, s):
    sr = score(s["raw"], s["target"])
    sc = score(s["clean"], s["target"])
    print(f"\n=== run {run} ===")
    print(f"  geometry floor      : {100*s['G'].mean():.2f}% of chip")
    print(f"  residual target     : {int(s['target'].sum())} px "
          f"({100*s['target'].mean():.3f}% of chip)")
    print(f"  RAW variance (k={K}) : {int(s['raw'].sum())} px  "
          f"T-prec {sr['precision']:.3f}  T-rec {sr['recall']:.3f}  IoU {sr['iou']:.3f}")
    print(f"  close_open (current): {int(s['clean'].sum())} px  "
          f"T-prec {sc['precision']:.3f}  T-rec {sc['recall']:.3f}  IoU {sc['iou']:.3f}")


def plot(run, s):
    bw = mcolors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    def show(a, m, t):
        a.imshow(m, cmap=bw); a.set_title(t, fontsize=11); a.axis("off")

    show(ax[0, 0], s["G"], f"geometry floor ({100*s['G'].mean():.1f}%)")
    # continuous score field, clipped for display
    zc = np.clip(s["z_field"], -3, 12)
    im = ax[0, 1].imshow(zc, cmap="inferno")
    ax[0, 1].set_title("low-variance score z\n(input to TV / graph-cut)", fontsize=11)
    ax[0, 1].axis("off"); fig.colorbar(im, ax=ax[0, 1], fraction=0.046)
    show(ax[0, 2], s["raw"], f"RAW outliers (z>{K})  {int(s['raw'].sum())} px\n"
                             "(point cloud for RANSAC / Hough)")
    show(ax[0, 3], s["clean"], f"close_open(raw) -- CURRENT\n{int(s['clean'].sum())} px")

    show(ax[1, 0], s["target"], f"residual target = human & ~geom\n"
                                f"{int(s['target'].sum())} px")
    show(ax[1, 1], s["human"], f"human_Mask ({100*s['human'].mean():.1f}%)")

    # raw vs target agreement
    a = np.ones((*s["target"].shape, 3))
    a[s["raw"] & s["target"]] = (0.0, 0.7, 0.0)
    a[s["raw"] & ~s["target"]] = (0.9, 0.0, 0.0)
    a[~s["raw"] & s["target"]] = (0.0, 0.3, 1.0)
    ax[1, 2].imshow(a); ax[1, 2].axis("off")
    ax[1, 2].set_title("raw vs target\ngreen=TP red=FP blue=FN", fontsize=11)

    a2 = np.ones((*s["target"].shape, 3))
    a2[s["clean"] & s["target"]] = (0.0, 0.7, 0.0)
    a2[s["clean"] & ~s["target"]] = (0.9, 0.0, 0.0)
    a2[~s["clean"] & s["target"]] = (0.0, 0.3, 1.0)
    ax[1, 3].imshow(a2); ax[1, 3].axis("off")
    ax[1, 3].set_title("close_open vs target\ngreen=TP red=FP blue=FN", fontsize=11)

    fig.suptitle(f"xppl1016922 run {run} -- geometry + variance substrate for "
                 f"shape-prior fitting", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, f"shape_prior_stage0_run{run:04d}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def main():
    for run in RUNS:
        s = stage0(run)
        report(run, s)
        for key in ("G", "cand", "target", "z_field", "raw", "clean"):
            np.save(os.path.join(OUT_DIR, f"{key}_run{run:04d}.npy"), s[key])
        plot(run, s)
    print(f"\n[arrays] {OUT_DIR}")


if __name__ == "__main__":
    main()
