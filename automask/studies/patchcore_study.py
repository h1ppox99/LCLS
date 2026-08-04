"""
studies/patchcore_study.py -- can an anomaly prior add to the production mask?

Scores the `patchcore` field stat (stats/patchcore.py) like any other detector in
the registry: robust-z field -> threshold at k -> mask regularizers, against the
human masks of runs 389/475.

The point of the study is what PatchCore is given BEFORE it runs. The trusted,
already-validated masks (geometry, calib, variance, hough_lines, asic_polish)
are not competition for it -- they are the definition of "normal" it lacks:

    known="floor"       geometry+calib only. The bank is then contaminated by
                        the real defects (they land in it, so their distance to
                        it is ~0) -- the failure mode this study started from.
    known="production"  the whole production mask. Removes those defects from
                        the bank with NO hand label, and the residual is what is
                        left to find. This is the configuration that makes an
                        unsupervised bank viable.
    known="human"       the hand mask (labelled upper bound / cross-run test).

`neutralize` goes one step further and paints the known-bad pixels out of the
*input*, so the backbone's receptive fields and the score blur never straddle a
dead band, and the contrast stretch is set by good pixels alone.

Because a detector can only ever contribute where production is already wrong,
the reported metrics are the honest ones:

    added        pixels the detector masks that production does not
    added prec   what fraction of those are in the human mask
    recovered    fraction of `human & ~production` it finds
    union IoU    IoU(production | patchcore) vs human -- does the pipeline improve?

Run (needs torch/timm, NOT the psana env):

    python -m automask.studies.patchcore_study
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import numpy as np

from automask.dataset import score
from automask.evaluation import EVAL_RUNS, load_sample
from automask.masking import production_pipeline
from automask.regularization.area_gate import AreaGateParams, area_gate
from automask.regularization.fill_holes import fill_holes
from automask.stats.base import threshold_stat
from automask.stats.patchcore import (PatchCoreParams, compute as patchcore_compute,
                                      known_mask, render)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "figures")

KS = np.round(np.arange(1.0, 8.01, 0.5), 2)

# label, PatchCoreParams overrides. bank_run is filled in per run for cross-run.
CONFIGS = [
    # The trusted masks are applied to the IMAGE first; `fill` decides how the
    # masked pixels are erased, which turns out to matter more than anything else.
    ("no fill (raw frame)", dict(known="production", fill="none"), True),
    ("fill=median (step edges)", dict(known="production", fill="median"), True),
    ("fill=nearest", dict(known="production", fill="nearest"), True),
    ("fill=texture", dict(known="production", fill="texture"), True),
    # Sharp: the residual production misses is a THIN scratch, and the default
    # regularization is built for blobs -- sigma=4 blur + a 200-px area gate
    # erase exactly that.
    ("fill=texture, sharp", dict(known="production", fill="texture",
                                 blur_sigma=1.0), False),
]


def pick(z, k, sample, regularize: bool = True):
    """Threshold the field like Detector.pick, then the production mask regs."""
    m = threshold_stat(z, k, "high") & sample.real
    if not regularize:
        return m
    return area_gate(fill_holes(m), min_area=AreaGateParams().min_area)


def contribution(m, prod, human):
    """What the detector adds on top of production, and whether it is right."""
    added = m & ~prod
    target = human & ~prod                       # what production is missing
    n_add = int(added.sum())
    return {"added": n_add,
            "added_prec": float((added & human).sum() / n_add) if n_add else 1.0,
            "recovered": float((added & target).sum() / target.sum())
                         if target.any() else 1.0,
            "union_iou": score(m | prod, human)["iou"]}


def figure(results, path, ks=KS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig = plt.figure(figsize=(23, 4.9 * n))
    gs = fig.add_gridspec(n, 5)
    for row, r in enumerate(results):
        s, prod, best = r["sample"], r["prod"], r["best"]
        added = best["mask"] & ~prod
        overlay = np.zeros(s.human.shape + (3,))
        overlay[prod] = (0.85, 0.85, 0.85)                     # production mask
        overlay[added & s.human] = (0.15, 0.60, 0.20)          # right addition
        overlay[added & ~s.human] = (0.85, 0.20, 0.15)         # wrong addition
        overlay[s.human & ~prod & ~added] = (0.20, 0.35, 0.85)  # still missed
        panels = [
            (r["cleaned"][0], "gray",
             f"run {s.run}: what PatchCore actually sees\n"
             f"(umean after the trusted masks, {best['label']})"),
            (np.clip(r["z_best"], -3, 12), "viridis",
             f"PatchCore anomaly field on the cleaned frame"),
            (prod, "magma", f"production pipeline   IoU {r['s_prod']['iou']:.3f}"),
            (overlay, None,
             f"grey = production | green/red = right/wrong additions\n"
             f"blue = still missed   union IoU {best['union_iou']:.3f}"),
        ]
        for col, (img, cmap, title) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[row, 4])
        for label, curve in r["curves"].items():
            ax.plot(ks, [c["union_iou"] for c in curve], marker="o", ms=3,
                    label=label)
        ax.axhline(r["s_prod"]["iou"], color="k", ls="--",
                   label=f"production alone ({r['s_prod']['iou']:.3f})")
        ax.set_xlabel("threshold k (robust-z)")
        ax.set_ylabel("IoU(production | patchcore) vs human")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_ylim(min(0.5, r["s_prod"]["iou"] - 0.1), 1.0)
        ax.set_title(f"run {s.run}: does adding PatchCore help?", fontsize=9)
    fig.suptitle("PatchCore given the trusted masks first: the production mask as "
                 "the definition of 'normal'", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"\nfigure -> {path}")


def main(runs: Sequence[int] = EVAL_RUNS, out: Optional[str] = None):
    pipe = production_pipeline("union")
    results = []
    for run in runs:
        other = [r for r in runs if r != run][0]
        sample = load_sample(run, features=pipe.features_needed() + ("umean",))
        prod = pipe.run(sample)
        s_prod = score(prod, sample.human)
        missing = int((sample.human & ~prod).sum())
        print(f"\n=== run {run} ===")
        print(f"  production: IoU {s_prod['iou']:.3f}  prec {s_prod['precision']:.3f}"
              f"  rec {s_prod['recall']:.3f}   -- {missing} px of the human mask "
              f"left to find")
        print(f"  {'bank / known mask':<36} {'k':>4} {'added':>8} {'prec':>6} "
              f"{'recov':>6} {'union IoU':>10}")
        curves, best = {}, None
        for label, over, reg in CONFIGS:
            over = dict(over)
            if over.get("bank_run") == "OTHER":
                over["bank_run"] = other
            z = patchcore_compute(sample, PatchCoreParams(**over))
            curve = [contribution(pick(z, float(k), sample, reg), prod, sample.human)
                     for k in KS]
            curves[label] = curve
            # The unconstrained argmax is degenerate: union IoU is maximal when
            # the detector adds NOTHING, so it selects an empty mask. Report the
            # best operating point that actually contributes something.
            adding = [j for j, c in enumerate(curve) if c["added"] > 0]
            if not adding:
                print(f"  {label:<36}    -        0      -      -  "
                      f"{s_prod['iou']:10.3f}  (never fires)")
                continue
            i = max(adding, key=lambda j: curve[j]["union_iou"])
            c, k_best = curve[i], float(KS[i])
            delta = c["union_iou"] - s_prod["iou"]
            print(f"  {label:<36} {k_best:4.1f} {c['added']:8d} "
                  f"{c['added_prec']:6.3f} {c['recovered']:6.3f} "
                  f"{c['union_iou']:10.3f}  ({delta:+.3f})")
            if best is None or c["union_iou"] > best["union_iou"]:
                best = {"label": label, "k": k_best, "z": z, "over": over,
                        "mask": pick(z, k_best, sample, reg), **c}
        bp = PatchCoreParams(**best["over"])
        cleaned = render(sample, known_mask(sample, bp.known), bp.fill, bp.seed)
        results.append({"sample": sample, "prod": prod, "s_prod": s_prod,
                        "curves": curves, "best": best, "z_best": best["z"],
                        "cleaned": cleaned})
    figure(results, out or os.path.join(OUT, "patchcore_study.png"))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(out=a.out)
