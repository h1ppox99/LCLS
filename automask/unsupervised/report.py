"""
unsupervised/report.py -- the production score card for a single mask.

`studies/metric_validation.py` answers "which of these masks is best?" by
z-scoring a panel of candidates against each other. Production has no panel: one
run arrives, one pipeline runs, and the question is "should I trust this mask?".
That needs each metric read against an ABSOLUTE reference rather than against
rivals, so this module pairs every metric with the value it takes when its own
null hypothesis is true:

    stab_*         1.0    identical masks from independent shots / jittered knobs
    azim_winrate   0.5    a coin flip against a size-matched random mask
    azim_gain      0.0    no better than dropping the same number of pixels
    event_gain     0.0    same, for the stationarity leak
    event_leak     1e-3   the chi2 test's own false-positive rate (P_ANOM), so a
                          perfect mask leaks exactly this, not zero
    compactness    ~1.0   defects are blobs; scattered singletons are noise
    plausible_frac 1.0    inside the plausible masked-fraction band

The verdict column is a coarse three-way flag, not a score: metrics disagree, and
the point of the card is to show WHERE a mask is weak (unreproducible? isotropy
not improved? still leaking non-stationary pixels?) rather than to collapse that
into one number a user would then over-trust.

Run:  python -m automask.unsupervised.report            # production, both runs
      python -m automask.unsupervised.report 475
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from automask.unsupervised.base import METRICS, Candidate, MetricContext, score_candidate

# metric -> (null value, "good" margin beyond the null in the improving direction)
REFERENCE = {
    "plausible_frac": (1.0, 0.0),
    "floor_containment": (1.0, 0.0),
    "compactness": (0.9, 0.05),
    "stab_noise": (1.0, 0.0),
    "stab_alt": (1.0, 0.0),
    "stab_time": (1.0, 0.0),
    "stab_hyper": (1.0, 0.0),
    "azim_excess": (float("nan"), float("nan")),   # scale is run-specific
    "azim_gain": (0.0, 0.0),
    "azim_winrate": (0.5, 0.05),
    # The nominal chi2 rate is 1e-3, but the measured leak on a good mask is
    # 4e-3 (run 389) to 6e-3 (run 475): real pixels are neither Poisson nor
    # independent, so the test over-rejects by ~5x. Judging against 1e-3 would
    # flag every mask WEAK, so the absolute level is reported without a verdict
    # and `event_gain` (leak relative to the size-matched control) carries the
    # decision instead.
    "event_leak": (float("nan"), float("nan")),
    "event_gain": (0.0, 0.0),
}

# How far below a "1.0 when the null holds" metric still counts as passing. Set
# from the validation panel: the production recipe reproduces at IoU ~0.96-0.99
# across independent shot halves, and every candidate that fell below ~0.90 there
# was over- or under-masking.
TOL = 0.90


def verdict(name: str, value: float) -> str:
    ref, margin = REFERENCE.get(name, (float("nan"), float("nan")))
    if not np.isfinite(value):
        return "n/a"
    if not np.isfinite(ref):
        return "--"
    better_high = METRICS[name].higher_is_better
    if ref == 1.0 and better_high:                 # reproducibility-style metric
        return "ok" if value >= TOL else "WEAK"
    if better_high:
        return "ok" if value >= ref + margin else "WEAK"
    return "ok" if value <= ref * 3 else "WEAK"    # leak-style: within 3x nominal


def score_card(
    pipeline, run: int, seed: int = 0,
    reductions=("std", "mean"), calibrations=("pedestals",),
):
    """Score one pipeline on one run with every label-free metric."""
    from automask.evaluation import load_sample
    from automask.unsupervised.stability import pipeline_jitter

    sample = load_sample(
        run, selection=pipeline.shot_selection, reductions=reductions,
        calibrations=calibrations,
    )
    ctx = MetricContext(sample=sample, seed=seed)
    cand = Candidate("pipeline", pipeline.run, jitter=pipeline_jitter(pipeline))
    row = score_candidate(cand, ctx)
    row["masked_frac"] = float(cand.mask(ctx).mean())
    return row


def print_card(row: dict, run: int) -> None:
    print(f"\n=== label-free score card, run {run} "
          f"({100*row['masked_frac']:.2f}% of the canvas masked) ===")
    print(f"{'metric':18s} {'tier':>4} {'value':>10} {'null':>10}   verdict   what it means")
    print("-" * 96)
    for name in sorted(METRICS, key=lambda n: (METRICS[n].tier, n)):
        v = row.get(name, float("nan"))
        ref = REFERENCE.get(name, (float("nan"),))[0]
        note = row.get(f"{name}__error", METRICS[name].doc)
        print(f"{name:18s} {METRICS[name].tier:>4} {v:10.4f} {ref:10.4f}   "
              f"{verdict(name, v):<8}  {note[:44]}")


def main(runs: Optional[Sequence[int]] = None):
    from automask.evaluation import EVAL_RUNS
    from automask.masking import production_pipeline

    pipe = production_pipeline()
    for run in (EVAL_RUNS if runs is None else runs):
        print_card(score_card(pipe, run), run)


if __name__ == "__main__":
    import sys
    main([int(a) for a in sys.argv[1:]] or None)
