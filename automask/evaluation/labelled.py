"""Reference-mask evaluation for fitting and held-out validation."""
from __future__ import annotations
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from automask.dataset import load_mask, score
from automask.image_store import ImageStore
from automask.io.read_xtc import available_xtc_runs
from automask.sample import Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_RUNS: Tuple[int, ...] = available_xtc_runs()
_SPLIT = len(ALL_RUNS) // 2
FIT_RUNS: Tuple[int, ...] = ALL_RUNS[:_SPLIT]
VALIDATION_RUNS: Tuple[int, ...] = ALL_RUNS[_SPLIT:]


def reference_mask(run: int) -> np.ndarray:
    """The hand-drawn target mask for `run` (bool, True == masked).

    Prefers a run-specific mask and falls back to the shared `human_Mask`.
    """
    per_run = os.path.join(PACKAGE, "data", "masks", f"human_Mask_run{run:04d}_asm.npy")
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
    """Evaluate a pipeline against reference masks on labelled runs.

    Returns {run: metrics, "mean": metrics}, where each metrics dict carries both
    the FULL-mask scores (pred vs reference) and the RESIDUAL scores (the pick
    beyond the floor vs reference & ~floor -- what the intensity channels must find).
    """
    runs = list(VALIDATION_RUNS if runs is None else runs)
    if not runs:
        raise ValueError("labelled evaluation needs at least one run")
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
