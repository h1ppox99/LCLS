"""
evaluation.py -- per-run Sample context + the evaluation loop.

`Sample` bundles the arrays a statistic reads for one run. Its selected-shot
images come from `ImageStore`, which serves a warm `.npy` cache when present and
otherwise reduces raw XTC frames. `load_sample` materializes only the reductions
and calibration constants a caller asks for. `evaluate` scores any
object exposing `.run(sample)`/`.floor(sample)` (a masking.Pipeline) across the
evaluation runs. `EVAL_RUNS` is the single place the evaluation set grows.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Sequence, Tuple

import numpy as np

from automask.dataset import load_image, load_mask, score
from automask.image_store import ImageStore, Reduction
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection

HERE = os.path.dirname(os.path.abspath(__file__))

# The evaluation set. Grows here, in one place, as more runs are frozen.
EVAL_RUNS: Tuple[int, ...] = (389, 475)

# psana pixel_status masks frozen per run into data/masks/.
_CALIB_MASK_BY_RUN = {389: "statusMask_run0389", 475: "statusMask_run0475"}
_REDUCTIONS = frozenset(("mean", "std", "median", "mad"))
_CALIBRATIONS = frozenset(("pedestals", "pixel_rms"))


@dataclass
class Sample:
    """Per-run masking context with reductions from one retained selection."""
    run: int
    sumimg: np.ndarray                  # calibrated run-sum image (lit), (1064, 1030)
    human: np.ndarray                   # reference target mask (bool, True == masked)
    calib: np.ndarray                   # psana pixel_status floor mask (bool)
    selection: ShotSelection = BEAM_ON_SELECTION
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    mad: Optional[np.ndarray] = None
    pedestals: Optional[np.ndarray] = None  # panel form, gain stage 0
    pixel_rms: Optional[np.ndarray] = None  # psana dark rms, gain stage 0

    @property
    def real(self) -> np.ndarray:
        """Pixels carrying a real value (sum != 0)."""
        return self.sumimg != 0

    @cached_property
    def center(self) -> Tuple[float, float]:
        """Frozen beam center in assembled (axis0, axis1) index order."""
        from automask.geometry import get_center
        return get_center(self.run)


def load_sample(
    run: int,
    selection: ShotSelection = BEAM_ON_SELECTION,
    reductions: Optional[Sequence[Reduction]] = None,
    calibrations: Optional[Sequence[str]] = None,
    store: Optional[ImageStore] = None,
) -> Sample:
    """Build a :class:`Sample` for `run`.

    Reductions share `selection`; calibration constants are loaded in panel form
    at gain stage zero. Omitting either requirement list loads all supported
    values, while evaluation passes a pipeline's exact requirements.
    """
    if not isinstance(selection, ShotSelection):
        raise TypeError("selection must be a ShotSelection")
    store = store or ImageStore()
    reductions = tuple(
        ("mean", "std", "median", "mad") if reductions is None else reductions
    )
    calibrations = tuple(
        ("pedestals", "pixel_rms") if calibrations is None else calibrations
    )
    unknown_reductions = set(reductions) - _REDUCTIONS
    unknown_calibrations = set(calibrations) - _CALIBRATIONS
    if unknown_reductions:
        raise ValueError(f"unsupported sample reductions: {sorted(unknown_reductions)}")
    if unknown_calibrations:
        raise ValueError(
            f"unsupported sample calibrations: {sorted(unknown_calibrations)}")
    sumimg = load_image(f"sum_calib_run{run:04d}").astype(np.float64)
    images = {
        reduction: store.reduce(run, selection, reduction).astype(np.float64)
        for reduction in reductions
    }
    constants = {
        constant: store.calibration(run, constant).astype(np.float64)
        for constant in calibrations
    }
    # Prefer a run-specific target (lab recipe re-run on this run); fall back to
    # the shared 475-built human_Mask.
    per_run = os.path.join(HERE, "data", "masks", f"human_Mask_run{run:04d}_asm.npy")
    human = (np.load(per_run).astype(bool) if os.path.exists(per_run)
             else load_mask("human_Mask"))
    try:
        calib = load_mask(_CALIB_MASK_BY_RUN[run])
    except KeyError:
        raise ValueError(
            f"no frozen calibration mask for run {run}; only "
            f"{sorted(_CALIB_MASK_BY_RUN)} are available") from None
    return Sample(
        run=run, sumimg=sumimg, human=human, calib=calib, selection=selection,
        **images, **constants,
    )


def evaluate(pipeline, runs: Optional[Sequence[int]] = None, verbose: bool = False):
    """Score `pipeline` (any object with `.run`/`.floor`) across `runs`.

    Returns {run: metrics, "mean": metrics}, where each metrics dict carries both
    the FULL-mask scores (floor | pred vs human) and the RESIDUAL scores (the pick
    beyond the floor vs human & ~floor -- what the intensity detectors must find).
    """
    runs = list(EVAL_RUNS if runs is None else runs)
    reductions = pipeline.reductions_needed()
    calibrations = pipeline.calibrations_needed()
    per_run = {}
    for run in runs:
        sample = load_sample(
            run, selection=pipeline.shot_selection, reductions=reductions,
            calibrations=calibrations,
        )
        floor = pipeline.floor(sample)
        pred = pipeline.run(sample)
        target = sample.human & ~floor
        full = score(pred, sample.human)
        resid = score(pred & ~floor, target)
        per_run[run] = {
            "iou": full["iou"], "precision": full["precision"], "recall": full["recall"],
            "masked_frac": float(pred.mean()),
            "residual_iou": resid["iou"],
            "residual_precision": resid["precision"],
            "residual_recall": resid["recall"],
            "floor_iou": score(floor, sample.human)["iou"],
        }
        if verbose:
            m = per_run[run]
            print(f"  run {run}: IoU {m['iou']:.3f}  prec {m['precision']:.3f}  "
                  f"rec {m['recall']:.3f}  ({100*m['masked_frac']:.2f}% masked)")

    keys = next(iter(per_run.values())).keys()
    mean = {k: float(np.mean([per_run[r][k] for r in runs])) for k in keys}
    return {**per_run, "mean": mean}
