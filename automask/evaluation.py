"""
evaluation.py -- per-run Sample context + the evaluation loop.

`Sample` bundles every array a statistic might read for one run; `load_sample`
builds it from the frozen numpy dataset (no psana). `evaluate` scores any object
exposing `.run(sample)` and `.floor(sample)` (a masking.Pipeline) across the
evaluation runs and aggregates the metrics. `EVAL_RUNS` is the single place the
evaluation set grows.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Sequence, Tuple

import numpy as np

from automask.dataset import load_image, load_mask, score

HERE = os.path.dirname(os.path.abspath(__file__))
_FEATURES = os.path.join(HERE, "data", "features")

# The evaluation set. Grows here, in one place, as more runs are frozen.
EVAL_RUNS: Tuple[int, ...] = (389, 475)

# psana pixel_status masks frozen per run into data/masks/.
_CALIB_MASK_BY_RUN = {389: "statusMask_run0389", 475: "statusMask_run0475"}


@dataclass
class Sample:
    """Per-run masking context. `mean` is a beam-OFF dark frame; `umean` is the
    genuine lit-beam per-pixel mean -- statistics needing scattering contrast
    (window_median, blackhat) must use `umean`, not `mean`."""
    run: int
    sumimg: np.ndarray      # calibrated run-sum image (lit), (1064, 1030)
    mean: np.ndarray        # beam-OFF dark/pedestal frame
    umean: np.ndarray       # per-shot-normalized lit-beam mean
    ustd: np.ndarray        # per-pixel std
    human: np.ndarray       # reference target mask (bool, True == masked)
    calib: np.ndarray       # psana pixel_status floor mask (bool)

    @property
    def real(self) -> np.ndarray:
        """Pixels carrying a real value (sum != 0)."""
        return self.sumimg != 0

    @cached_property
    def center(self) -> Tuple[float, float]:
        """Beam center in (axis0, axis1) index order. Lazily read from the run's
        small-data HDF5 (the only h5py dependency, needed by the radial_median /
        azimuthal_sigma stats); cached so the core numpy-only stats never pay it."""
        from automask.geometry import get_center
        return get_center(self.run)


def load_sample(run: int) -> Sample:
    """Build a :class:`Sample` for `run` from the frozen numpy dataset."""
    sumimg = load_image(f"sum_calib_run{run:04d}").astype(np.float64)
    mean = np.load(os.path.join(_FEATURES, f"mean_run{run:04d}_asm.npy")).astype(np.float64)
    ustd = np.load(os.path.join(_FEATURES, f"ustd_run{run:04d}_asm.npy")).astype(np.float64)
    umean = np.load(os.path.join(_FEATURES, f"umean_run{run:04d}_asm.npy")).astype(np.float64)
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
    return Sample(run=run, sumimg=sumimg, mean=mean, umean=umean, ustd=ustd,
                  human=human, calib=calib)


def evaluate(pipeline, runs: Optional[Sequence[int]] = None, verbose: bool = False):
    """Score `pipeline` (any object with `.run`/`.floor`) across `runs`.

    Returns {run: metrics, "mean": metrics}, where each metrics dict carries both
    the FULL-mask scores (floor | pred vs human) and the RESIDUAL scores (the pick
    beyond the floor vs human & ~floor -- what the intensity detectors must find).
    """
    runs = list(EVAL_RUNS if runs is None else runs)
    per_run = {}
    for run in runs:
        sample = load_sample(run)
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
