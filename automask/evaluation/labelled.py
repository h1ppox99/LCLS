"""Reference-mask evaluation for fitting and held-out validation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from automask.evaluation.dataset import reference_mask_path, reference_runs, score
from automask.sample.image_store import ImageStore
from automask.sample import Sample
from automask.selection.presets import BEAM_ON_SELECTION
from automask.selection.shot_selection import ShotSelection


def _partition_runs(runs: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    runs = tuple(runs)
    split = len(runs) // 2
    return runs[:split], runs[split:]


ALL_RUNS: Tuple[int, ...] = reference_runs()
FIT_RUNS, VALIDATION_RUNS = _partition_runs(ALL_RUNS)


def reference_mask(run: int) -> np.ndarray:
    """The hand-drawn target mask for `run` (bool, True == masked)."""
    path = reference_mask_path(run)
    if not path.exists():
        available = sorted(item.name for item in path.parent.glob("*.npy"))
        raise FileNotFoundError(f"{path}\navailable: {available}")
    return np.load(path).astype(bool)


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
            "iou": full["iou"],
            "precision": full["precision"],
            "recall": full["recall"],
            "masked_frac": float(pred.mean()),
            "residual_iou": resid["iou"],
            "residual_precision": resid["precision"],
            "residual_recall": resid["recall"],
            "floor_iou": score(floor, human)["iou"],
        }
        if verbose:
            m = per_run[run]
            print(
                f"  run {run}: IoU {m['iou']:.3f}  prec {m['precision']:.3f}  "
                f"rec {m['recall']:.3f}  ({100 * m['masked_frac']:.2f}% masked)"
            )

    keys = next(iter(per_run.values())).keys()
    mean = {k: float(np.mean([per_run[r][k] for r in runs])) for k in keys}
    return {**per_run, "mean": mean}
