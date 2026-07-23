"""
exp_combiners_ablation.py -- Direction 3: paired combiner comparison + ablation.

All variants are scored on the SAME seeded corruptions and the same real masks,
so differences are attributable to the pipeline. Reports real-mask IoU/prec/rec
and per-artifact synthetic recall (mean ± std), both runs.

    source psana_env.sh
    python -m automask.studies.exploration.exp_combiners_ablation
"""
from __future__ import annotations

import numpy as np

from automask.masking import Detector, Pipeline, production_pipeline
from automask.stats.variance import VarianceParams
from automask.stats.window_median import WindowMedianParams
from automask.stats.blackhat import BlackhatParams
from automask.regularization.tv import TVParams
from automask.regularization.pad import PadParams
from automask.combine.weighted_sum import WeightedSumParams
from automask.studies.exploration.harness import synth, agg, real

RUNS = (389, 475)
SEED, NPER, ROT = 0, 5, (0, 180)
ART = ("dead_pixels", "hot_pixels", "bad_column", "beamstop", "beamstop_small", "streak")


def _detectors():
    return [
        Detector("variance", VarianceParams(k=3.5, mode="low"),
                 field_reg="tv", field_reg_params=TVParams(4.0)),
        Detector("window_median", WindowMedianParams(win=21, k=5.0, mode="low"),
                 field_reg="tv", field_reg_params=TVParams(1.0),
                 mask_reg="pad", mask_reg_params=PadParams(2)),
        Detector("blackhat", BlackhatParams(radius=5, k=6.0, mode="high"),
                 field_reg="tv", field_reg_params=TVParams(1.0),
                 mask_reg="pad", mask_reg_params=PadParams(2)),
    ]


FLOOR = ["geometry", "calib", "dead_holes", "stuck"]   # the validated best floor (D1)


def _pipe(dets, combiner="union", cp=None):
    return Pipeline(dets, floor_stats=FLOOR, combiner=combiner, combiner_params=cp)


def variants():
    d = _detectors()
    v = {
        "union": _pipe(d, "union"),
        "weighted_sum": _pipe(d, "weighted_sum", WeightedSumParams(k=3.5, pad=2)),
        "mahalanobis": _pipe(d, "mahalanobis"),
        # ablations (union): drop one detector at a time
        "abl_no_variance": _pipe([d[1], d[2]], "union"),
        "abl_no_window": _pipe([d[0], d[2]], "union"),
        "abl_no_blackhat": _pipe([d[0], d[1]], "union"),
        "only_variance": _pipe([d[0]], "union"),
        "floor_only": _pipe([], "union"),
    }
    return v


def main():
    print(f"=== Direction 3: combiners + ablation | runs {RUNS} "
          f"seed={SEED} n_per={NPER} rot={ROT} ===\n")
    hdr = f"{'variant':16s} | {'real IoU':>8s} {'prec':>6s} {'rec':>6s} | " + \
          " ".join(f"{a[:8]:>8s}" for a in ART)
    print(hdr); print("-" * len(hdr))
    for name, pipe in variants().items():
        r = real(pipe, runs=RUNS)["mean"]
        rows = synth(pipe, runs=RUNS, seed=SEED, n_per=NPER, rotations=ROT,
                     tag=f"c_{name}")
        a = agg(rows, "recall")
        rec = " ".join(f"{a.get(art,(float('nan'),))[0]:8.3f}" for art in ART)
        print(f"{name:16s} | {r['iou']:8.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} | {rec}")


if __name__ == "__main__":
    main()
