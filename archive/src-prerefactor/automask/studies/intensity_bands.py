#!/usr/bin/env python3
"""
intensity_bands.py -- does the masking pipeline behave differently per shot intensity?

`methods.py` builds its feature maps (mean, ustd) from frames of EVERY intensity,
so a run-sum mixes near-dark shots with the brightest ones.  The premise here: a
mask found on all shots catches anomalies "at every scale", but an intensity-
RESTRICTED pass may expose defects that only appear at one illumination level --
a dead pixel is invisible in a dim frame, a saturating one only misbehaves bright.

This script runs the UNCHANGED pipeline (G | A | C from methods.py, same
parameters) on four feature sets and scores them all against `human_Mask`:

    all      the existing all-intensity maps        (the published IoU 0.747)
    all_raw  all intensities, RAW mean -- the matched control, see below
    p03_10   only shots in the 3-10th percentile of ipm2  (run 475: ipm2 15..190)
    p90_97   only shots in the 90-97th percentile         (run 475: ipm2 14799..23553)

The band feature maps come from `../../extract_band_features.py` (one psana pass).

WHY `all_raw` EXISTS -- the published baseline is not a like-for-like control.
`methods.py` feeds `method_blackhat` the mean from `pixel_stats.py`, which is the
photons-DROPPED sum / N: a sparse, thresholded quantity whose median over live
pixels is exactly 0.0 and which shows no diffraction rings.  The band means here
are plain calibrated means (median ~0.24, rings clearly visible) -- a different
quantity, not just different shots.  Comparing a band against the published number
would therefore confound "which shots" with "which estimator".  `all_raw` uses the
800-frame raw `umean`/`ustd` from `../../read_normalized_median.py`: same estimator
as the bands, all intensities.  **`all_raw` is the column the bands must be judged
against.**  (`ustd` is raw in every column, so A was never affected -- only C.)

The geometry floor G is recomputed per band from that band's own sum image, so
each column is a genuine end-to-end run of the pipeline rather than a shared
floor -- G is intensity-free by construction, so any difference between the
columns' G is itself worth seeing (it is reported).

Outputs:
  images/intensity_bands/pipeline_by_band_run0475.png   (per-band 4-panel + agreement)
  images/intensity_bands/features_by_band_run0475.png   (mean/ustd maps that drive it)
  results/combo_run0475_p03_10.npy, combo_run0475_p90_97.npy

Run:  python intensity_bands.py     (from src/automask/, numpy only)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import load_image, load_mask, score
from methods import geometry_mask, method_variance, method_blackhat, _agree

FEATURES = os.path.join(AUTOMASK, "data", "features")
IMAGES = os.path.join(AUTOMASK, "outputs", "figures", "intensity_bands")
RESULTS = os.path.join(AUTOMASK, "outputs", "masks")
os.makedirs(IMAGES, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
RUN = 475
BANDS = ["p03_10", "p90_97"]
COLUMNS = ["all", "all_raw"] + BANDS


def load_band(run, tag):
    """Feature set for one column: (sum, mean, ustd), assembled.

    "all"     -> the published maps (DROPPED mean -- see the module docstring)
    "all_raw" -> 800-frame raw umean/ustd, all intensities: the matched control
    "pNN_MM"  -> one intensity band from extract_band_features.py
    """
    sumimg = load_image(f"sum_calib_run{run:04d}").astype(np.float64)
    feat = lambda n: np.load(os.path.join(FEATURES, f"{n}_run{run:04d}_asm.npy")).astype(np.float64)
    if tag == "all":
        return sumimg, feat("mean"), feat("ustd")
    if tag == "all_raw":
        # umean = the 800-frame raw all-intensity mean, frozen from
        # ../h5/normalized_median_run0475.h5 (umean_asm) to keep this numpy-only.
        return sumimg, feat("umean"), feat("ustd")
    f = lambda n: np.load(os.path.join(FEATURES, f"{n}_run{run:04d}_{tag}_asm.npy")).astype(np.float64)
    return f("sum"), f("mean"), f("ustd")


def band_meta(run):
    p = os.path.join(FEATURES, f"band_features_run{run:04d}.json")
    with open(p) as fh:
        return json.load(fh)["bands"]


def run_pipeline(sumimg, mean, ustd):
    """The methods.py pipeline, verbatim: geometry floor then variance + black-hat."""
    real = sumimg != 0
    G = geometry_mask(real)
    A = method_variance(mean, ustd, real)
    C = method_blackhat(mean, real)
    return {"real": real, "G": G, "A": A, "C": C, "combo": G | A | C}


def describe(tag, meta):
    """Human-readable band label: "p03_10" -> its percentile / ipm2 / shot count."""
    if tag == "all":
        return "all intensities\n(published, dropped mean)"
    if tag == "all_raw":
        return "all intensities, raw mean\n800 shots [CONTROL]"
    key = "_".join(str(int(x)) for x in tag[1:].split("_"))   # "p03_10" -> "3_10"
    b = meta[key]
    lo, hi = b["ipm2_edges"]
    return (f"{b['percentile'][0]}-{b['percentile'][1]}% intensity\n"
            f"ipm2 {lo:.0f}-{hi:.0f}, {b['n_frames']} shots")


# ==========================================================================
#  figures
# ==========================================================================
def plot_features(sets, meta, run):
    """The mean and ustd maps each column masks on -- why the scores differ."""
    tags = list(sets)
    fig, ax = plt.subplots(2, len(tags), figsize=(5.4 * len(tags), 10))
    for j, tag in enumerate(tags):
        sumimg, mean, ustd = sets[tag]["features"]
        real = sets[tag]["real"]
        for i, (m, name) in enumerate([(mean, "mean (ADU/px)"), (ustd, "ustd (ADU/px)")]):
            v = m[real]
            im = ax[i, j].imshow(m, vmin=np.percentile(v, 1), vmax=np.percentile(v, 99),
                                 cmap="viridis")
            ax[i, j].set_title(f"{describe(tag, meta)}\n{name}", fontsize=10)
            ax[i, j].axis("off")
            fig.colorbar(im, ax=ax[i, j], fraction=0.046)
    fig.suptitle(f"xppl1016922 run {run} — pipeline input features, per intensity band "
                 f"(each panel autoscaled 1-99%)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(IMAGES, f"features_by_band_run{run:04d}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


def plot_pipeline_by_band(sets, human, meta, run):
    """One column per band: A, C, the combo, and the agreement vs human_Mask."""
    tags = list(sets)
    bw = mcolors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(4, len(tags), figsize=(5.4 * len(tags), 20))

    for j, tag in enumerate(tags):
        r = sets[tag]
        T = human & ~r["G"]
        Ar, Cr = r["A"] & ~r["G"], r["C"] & ~r["G"]
        sa, sc = score(Ar, T), score(Cr, T)
        sk = score(r["combo"], human)

        def show(a, m, title):
            a.imshow(m, cmap=bw); a.set_title(title, fontsize=10); a.axis("off")

        show(ax[0, j], Ar, f"{describe(tag, meta)}\nA variance (A&~G), {100*Ar.mean():.2f}%\n"
                           f"T-prec={sa['precision']:.2f} T-rec={sa['recall']:.3f}")
        show(ax[1, j], Cr, f"C black-hat (C&~G), {100*Cr.mean():.2f}%\n"
                           f"T-prec={sc['precision']:.2f} T-rec={sc['recall']:.3f}")
        show(ax[2, j], r["combo"], f"G ∪ A ∪ C ({100*r['combo'].mean():.1f}%)\n"
                                   f"IoU={sk['iou']:.3f} prec={sk['precision']:.2f} "
                                   f"rec={sk['recall']:.2f}")
        ax[3, j].imshow(_agree(r["combo"], human))
        ax[3, j].set_title(f"vs human_Mask — green=TP red=FP blue=FN\n"
                           f"IoU={sk['iou']:.3f}", fontsize=10)
        ax[3, j].axis("off")

    fig.suptitle(f"xppl1016922 run {run} — the SAME pipeline (G|A|C) run per intensity band",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(IMAGES, f"pipeline_by_band_run{run:04d}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


# ==========================================================================
#  main
# ==========================================================================
def main():
    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    meta = band_meta(RUN)
    human = load_mask("human_Mask")

    sets = {}
    for tag in COLUMNS:
        feats = load_band(RUN, tag)
        r = run_pipeline(*feats)
        r["features"] = feats
        sets[tag] = r

    # --- scores -----------------------------------------------------------
    print(f"\n=== G|A|C vs human_Mask, per intensity band (run {RUN}) ===")
    print("(judge the bands against all_raw -- the matched-estimator control)")
    print(f"{'band':30s} | {'masked%':>7s} | {'IoU':>6s} {'prec':>6s} {'rec':>6s} "
          f"| {'A add%':>6s} {'A T-prec':>8s} | {'C add%':>6s} {'C T-prec':>8s}")
    print("-" * 108)
    for tag, r in sets.items():
        T = human & ~r["G"]
        Ar, Cr = r["A"] & ~r["G"], r["C"] & ~r["G"]
        sk, sa, sc = score(r["combo"], human), score(Ar, T), score(Cr, T)
        name = describe(tag, meta).replace("\n", " ")[:30]
        print(f"{name:30s} | {100*r['combo'].mean():6.2f}% | {sk['iou']:6.3f} "
              f"{sk['precision']:6.3f} {sk['recall']:6.3f} | {100*Ar.mean():5.2f}% "
              f"{sa['precision']:8.3f} | {100*Cr.mean():5.2f}% {sc['precision']:8.3f}")

    # geometry floor should be intensity-free -- verify it actually is
    print("\n=== geometry floor G per band (should be intensity-independent) ===")
    G0 = sets["all"]["G"]
    for tag, r in sets.items():
        d = int((r["G"] ^ G0).sum())
        s = score(r["G"], human)
        print(f"  {tag:8s}: {100*r['G'].mean():5.2f}% masked  IoU {s['iou']:.3f}  "
              f"prec {s['precision']:.3f}  ({d} px differ from the all-shots G)")

    for tag in BANDS:
        p = os.path.join(RESULTS, f"combo_run{RUN:04d}_{tag}.npy")
        np.save(p, sets[tag]["combo"])
        print(f"[saved] {p}")

    plot_features(sets, meta, RUN)
    plot_pipeline_by_band(sets, human, meta, RUN)


if __name__ == "__main__":
    main()
