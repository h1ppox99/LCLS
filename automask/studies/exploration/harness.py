"""
studies/exploration/harness.py -- statistically-honest comparison harness.

Thin wrappers over the existing evaluators so exploration experiments report
spread, not just means, and compare pipelines on IDENTICAL corruptions (paired).

    from automask.studies.exploration.harness import synth, agg, real, ARTS_ALL

`synth` runs the pipeline-mode synthetic suite for a given Pipeline and returns
per-example rows; `agg` reduces rows to mean/std/n per artifact type. `real`
scores a pipeline on the true human masks. Everything is deterministic in the
seed, so two pipelines evaluated with the same (runs, seed, rotations, n_per)
see the same injected artifacts -> differences are attributable to the pipeline.
"""
from __future__ import annotations

import csv
import os
import tempfile
from collections import defaultdict

import numpy as np

from automask.synthetic.evaluate import evaluate_config

# Full physically-grounded artifact suite (extended + point/column defects).
ARTS_ALL = {
    "streak": {"angle_deg": [0.0, 180.0], "width": [1.5, 3.5],
               "amplitude_sigma": [6.0, 12.0], "band_sigmas": 1.0},
    "beamstop": {"shape": "random", "radius": [30.0, 100.0], "axis_ratio": [0.6, 1.6],
                 "transmission": [0.05, 0.30], "softness": 0.15},
    "beamstop_small": {"shape": "random", "radius": [15.0, 50.0], "axis_ratio": [0.6, 1.6],
                       "transmission": [0.05, 0.30], "softness": 0.15},
    "dead_pixels": {"n": [80, 400], "cluster": 1},
    "hot_pixels": {"n": [80, 400], "cluster": 1, "amplitude_sigma": [8.0, 20.0]},
    "bad_column": {"n_cols": [1, 4], "length_frac": [0.3, 1.0], "polarity": "dead"},
}

_METRICS = ["precision", "recall", "f1", "iou", "fpr", "masked_frac"]


def synth(pipeline, runs=(389, 475), artifacts=None, seed=0, n_per=5,
          rotations=(0, 180), tag="run"):
    """Run the synthetic pipeline suite for `pipeline`; return per-example rows."""
    out_dir = os.path.join(tempfile.gettempdir(), f"expl_{tag}")
    cfg = {
        "mode": "pipeline", "runs": list(runs), "combiner": "union",
        "output_dir": out_dir, "seed": seed, "n_per_artifact": n_per,
        "rotations": list(rotations), "save_figures": False,
        "artifacts": artifacts or ARTS_ALL,
    }
    evaluate_config(cfg, pipeline=pipeline)
    rows = []
    with open(os.path.join(out_dir, "results.csv")) as f:
        for r in csv.DictReader(f):
            for k in _METRICS + ["injected_px"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows


def agg(rows, metric="recall"):
    """mean/std/n of `metric` per artifact type (+ overall)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["artifact_type"]].append(r[metric])
    out = {}
    for name, vals in groups.items():
        out[name] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    allv = [r[metric] for r in rows]
    out["overall"] = (float(np.mean(allv)), float(np.std(allv)), len(allv))
    return out


def real(pipeline, runs=(389, 475)):
    """True-mask evaluation (IoU/prec/rec) via the standard evaluate loop."""
    from automask.evaluation import evaluate
    return evaluate(pipeline, runs=list(runs))


def synth_delta(pipeline, runs=(389, 475), artifacts=None, seed=0, n_per=5,
                rotations=(0, 180)):
    """Incremental (delta) synthetic scoring: de-confounds precision (finding M1).

    For each example, score the ARTIFACT-ATTRIBUTABLE mask -- the pixels the
    pipeline masks on the corrupted Sample but NOT on the clean one
    (`pred_corrupted & ~pred_clean`) -- against the injected truth. This removes
    the pipeline's always-on baseline masking from the false-positive count, so
    precision reflects the artifact response rather than a constant. Returns rows
    with standard recall and both standard and delta precision/IoU."""
    from automask.evaluation import load_sample
    from automask.synthetic.sample_adapter import corrupt_sample, rotate_sample
    from automask.synthetic.evaluate import _sample_params
    from automask.synthetic.metrics import masking_metrics
    arts = artifacts or ARTS_ALL
    rows = []
    for si, run in enumerate(runs):
        base = load_sample(run)
        for ri, deg in enumerate(rotations):
            rs = rotate_sample(base, deg)
            region = ~rs.human
            pred_clean = pipeline.run(rs).astype(bool)      # baseline masking, no artifact
            for ti, (name, acfg) in enumerate(arts.items()):
                for i in range(n_per):
                    s_i = seed + si * 1_000_000 + ri * 100_000 + ti * 10_000 + i
                    rng = np.random.default_rng(s_i)
                    params = _sample_params(rng, acfg)
                    cs, inj = corrupt_sample(rs, name, rng, params)
                    pred = pipeline.run(cs).astype(bool)
                    m = masking_metrics(pred, inj, region)
                    md = masking_metrics(pred & ~pred_clean, inj, region)
                    rows.append({"artifact_type": name, "recall": m["recall"],
                                 "precision": m["precision"], "iou": m["iou"],
                                 "precision_delta": md["precision"], "iou_delta": md["iou"]})
    return rows


def sample_custom(run, selection):
    """A Sample whose umean/ustd come from a CUSTOM ShotSelection (direction 2).

    sumimg/human/calib and the dark `mean` are the catalogue values; only the lit
    features are recomputed (from the warm cache or XTC) for `selection`. Use with
    `score_sample` -- NOT `evaluate`, which would re-pull the catalogue features."""
    from dataclasses import replace
    from automask.features import FeatureSpec, FeatureStore
    from automask.evaluation import load_sample
    store = FeatureStore()
    umean = store.get(run, FeatureSpec("umean", "mean", selection)).astype(float)
    ustd = store.get(run, FeatureSpec("ustd", "std", selection)).astype(float)
    base = load_sample(run, features=["mean"])   # dark + sumimg/human/calib
    return replace(base, umean=umean, ustd=ustd)


def score_sample(pipeline, sample):
    """Real-mask IoU/precision/recall of `pipeline` on one (possibly custom) Sample."""
    from automask.dataset import score
    pred = pipeline.run(sample)
    return score(pred, sample.human)


def show(rows, title="", metric="recall"):
    a = agg(rows, metric)
    print(f"\n{title}  [{metric}]  (mean ± std, n)")
    for name in ["streak", "beamstop", "beamstop_small", "dead_pixels",
                 "hot_pixels", "bad_column", "original", "overall"]:
        if name in a:
            m, s, n = a[name]
            print(f"  {name:16s} {m:6.3f} ± {s:5.3f}  (n={n})")
    return a
