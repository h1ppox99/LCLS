"""
evaluation.py -- scoring a pipeline against the hand-drawn reference masks.

This is the ONLY place a frozen `.npy` is still read, and deliberately so: a
human reference mask is a measurement someone made once, not something psana can
recompute. Everything a pipeline *consumes* now comes from the run itself (see
`automask.sample.Sample.from_store`); everything it is *judged against* lives
here.

`evaluate` scores any object exposing `.run(sample)`/`.floor(sample)` (a
masking.Pipeline) across the evaluation runs. `EVAL_RUNS` is the single place the
evaluation set grows.
"""
from __future__ import annotations
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from automask.dataset import load_mask, score
from automask.image_store import ImageStore
from automask.sample import Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection

HERE = os.path.dirname(os.path.abspath(__file__))

# The evaluation set. Grows here, in one place, as more runs are frozen.
EVAL_RUNS: Tuple[int, ...] = (389, 475)


def reference_mask(run: int) -> np.ndarray:
    """The hand-drawn target mask for `run` (bool, True == masked).

    Prefers a run-specific mask (the lab recipe re-run on this run) and falls
    back to the shared run-475 `human_Mask`, so run-389 scores are provisional.
    """
    per_run = os.path.join(HERE, "data", "masks", f"human_Mask_run{run:04d}_asm.npy")
    if os.path.exists(per_run):
        return np.load(per_run).astype(bool)
    return load_mask("human_Mask")


def evaluate(
    pipeline,
    runs: Optional[Sequence[int]] = None,
    selection: ShotSelection = BEAM_ON_SELECTION,
    store: Optional[ImageStore] = None,
    verbose: bool = False,
):
    """Score `pipeline` (any object with `.run`/`.floor`/`.needs`) across `runs`.

    Returns {run: metrics, "mean": metrics}, where each metrics dict carries both
    the FULL-mask scores (pred vs reference) and the RESIDUAL scores (the pick
    beyond the floor vs reference & ~floor -- what the intensity channels must find).
    """
    runs = list(EVAL_RUNS if runs is None else runs)
    store = store or ImageStore()
    per_run = {}
    for run in runs:
        sample = Sample.from_store(run, selection, pipeline.needs(), store=store)
        human = reference_mask(run)
        floor = pipeline.floor(sample)
        pred = pipeline.run(sample)
        target = human & ~floor
        full = score(pred, human)
        resid = score(pred & ~floor, target)
        per_run[run] = {
            "iou": full["iou"], "precision": full["precision"], "recall": full["recall"],
            "masked_frac": float(pred.mean()),
            "residual_iou": resid["iou"],
            "residual_precision": resid["precision"],
            "residual_recall": resid["recall"],
            "floor_iou": score(floor, human)["iou"],
        }
        if verbose:
            m = per_run[run]
            print(f"  run {run}: IoU {m['iou']:.3f}  prec {m['precision']:.3f}  "
                  f"rec {m['recall']:.3f}  ({100*m['masked_frac']:.2f}% masked)")

    keys = next(iter(per_run.values())).keys()
    mean = {k: float(np.mean([per_run[r][k] for r in runs])) for k in keys}
    return {**per_run, "mean": mean}


def report(run: int, selection: ShotSelection = BEAM_ON_SELECTION, store=None):
    """Per-channel + combined scores for one run, with agreement figures."""
    from automask import viz
    from automask.masking import production_pipeline

    pipe = production_pipeline("union")
    pipe_sum = production_pipeline("weighted_sum")
    sample = Sample.from_store(run, selection, pipe.needs(), store=store)
    human = reference_mask(run)

    floor = pipe.floor(sample)
    target = human & ~floor
    sf = score(floor, human)
    print(f"=== floor ({'+'.join(c.label for c in pipe.floor_channels)}), run {run} ===")
    print(f"  floor vs human : {100*floor.mean():.2f}% masked  IoU {sf['iou']:.3f}  "
          f"prec {sf['precision']:.3f}  rec {sf['recall']:.3f}")
    print(f"  residual target = human & ~floor : {int(target.sum())} px to find\n")

    picks = {c.label: c.pick(sample) for c in pipe.evidence_channels}
    combo, combo_sum = pipe.run(sample), pipe_sum.run(sample)

    print(f"{'channel':16s} | {'added%':>6s} {'T-prec':>6s} {'T-rec':>6s} "
          f"| {'IoU':>7s} {'prec':>6s} {'rec':>6s}")
    print("-" * 70)
    for name, M in {**picks, "combo": combo, "combo_sum": combo_sum}.items():
        Mo = M & ~floor
        st, sc = score(Mo, target), score(floor | M, human)
        print(f"{name:16s} | {100*Mo.mean():5.2f}% {st['precision']:6.3f} "
              f"{st['recall']:6.3f} | {sc['iou']:7.3f} {sc['precision']:6.3f} "
              f"{sc['recall']:6.3f}")
    print(f"\n(reference: floor alone -> IoU {sf['iou']:.3f})")

    out_dir = os.path.join(HERE, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    for name, M in {**picks, "combo": combo, "combo_sum": combo_sum}.items():
        out = os.path.join(out_dir, f"{name}_run{run:04d}.png")
        viz.save_agreement(floor | M, floor, human, run, out,
                           title=f"run {run} — {name}")
        print(f"[figure] {out}")
    panels = os.path.join(out_dir, f"panels_run{run:04d}.png")
    viz.channel_panels(pipe, sample, out=panels)
    print(f"[figure] {panels}")


def main(runs: Optional[Sequence[int]] = None):
    for run in (EVAL_RUNS if runs is None else runs):
        report(run)
        print()


if __name__ == "__main__":
    main()
