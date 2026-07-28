"""
evaluation.py -- per-run Sample context + the evaluation loop.

`Sample` bundles the arrays a statistic reads for one run. Its shot-selection
features (`mean`/`umean`/`ustd`) come from the FeatureStore, which serves a warm
`.npy` cache when present and otherwise computes the feature from raw XTC (see
`automask.features`). `load_sample` materializes only the features a caller asks
for -- `evaluate` passes exactly what the pipeline's stats declare via `needs`,
so the masking strategy drives which features are built. `evaluate` scores any
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
from automask.features import FEATURES, FeatureStore, get_spec

HERE = os.path.dirname(os.path.abspath(__file__))

# The evaluation set. Grows here, in one place, as more runs are frozen.
EVAL_RUNS: Tuple[int, ...] = (389, 475)

# psana pixel_status masks frozen per run into data/masks/.
_CALIB_MASK_BY_RUN = {389: "statusMask_run0389", 475: "statusMask_run0475"}


@dataclass
class Sample:
    """Per-run masking context. The feature fields are shot-selection reductions
    from the FeatureStore (`automask.features`): `mean` is a beam-OFF dark frame,
    `umean` the lit-beam per-pixel mean, `ustd` the lit-beam per-pixel std --
    statistics needing scattering contrast (blackhat) must use
    `umean`, not `mean`. Feature fields are `None` when a caller did not request
    them (see `load_sample(features=...)`); a stat that reads one it wasn't given
    gets a clear error rather than silent garbage."""
    run: int
    sumimg: np.ndarray                  # calibrated run-sum image (lit), (1064, 1030)
    human: np.ndarray                   # reference target mask (bool, True == masked)
    calib: np.ndarray                   # psana pixel_status floor mask (bool)
    mean: Optional[np.ndarray] = None   # beam-OFF dark/pedestal frame
    umean: Optional[np.ndarray] = None  # lit-beam per-pixel mean
    ustd: Optional[np.ndarray] = None   # lit-beam per-pixel std
    umad: Optional[np.ndarray] = None   # lit-beam per-pixel MAD (robust std)
    # Calibration constants, PANEL form (2, 512, 1024) -- not assembled space.
    # These describe the detector, not this run's beam, so they are shared by
    # every run in the same calibration epoch.
    pedestal: Optional[np.ndarray] = None   # psana pedestals, gain stage 0
    pixel_rms: Optional[np.ndarray] = None  # psana dark rms, gain stage 0

    @property
    def real(self) -> np.ndarray:
        """Pixels carrying a real value (sum != 0)."""
        return self.sumimg != 0

    @cached_property
    def center(self) -> Tuple[float, float]:
        """Beam center in (axis0, axis1) index order. Lazily read from the run's
        small-data HDF5 (the only h5py dependency, needed by the sigma_clipping
        stat); cached so the core numpy-only stats never pay it."""
        from automask.geometry import get_center
        return get_center(self.run)


def load_sample(run: int, features: Optional[Sequence[str]] = None,
                store: Optional[FeatureStore] = None) -> Sample:
    """Build a :class:`Sample` for `run`.

    `features` names the shot-selection features to materialize (default: the
    whole catalogue). Each is resolved through the FeatureStore -- served from
    the warm cache if present, else computed from XTC. Pass the subset a pipeline
    actually needs (see `Pipeline.features_needed`) to avoid building the rest.
    """
    store = store or FeatureStore()
    if features is None:
        features = tuple(FEATURES)
    sumimg = load_image(f"sum_calib_run{run:04d}").astype(np.float64)
    # Each spec declares the layout it is served in: intensity reductions come in
    # assembled space, calib constants in native panel geometry (see FeatureSpec.form).
    feats = {name: store.get(run, get_spec(name), form=get_spec(name).form)
             .astype(np.float64) for name in features}
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
    return Sample(run=run, sumimg=sumimg, human=human, calib=calib, **feats)


def evaluate(pipeline, runs: Optional[Sequence[int]] = None, verbose: bool = False):
    """Score `pipeline` (any object with `.run`/`.floor`) across `runs`.

    Returns {run: metrics, "mean": metrics}, where each metrics dict carries both
    the FULL-mask scores (floor | pred vs human) and the RESIDUAL scores (the pick
    beyond the floor vs human & ~floor -- what the intensity detectors must find).
    """
    runs = list(EVAL_RUNS if runs is None else runs)
    # Materialize only the features this pipeline's stats declare (step 2 of the
    # hierarchy: the masking strategy decides which features get built).
    needed = pipeline.features_needed() if hasattr(pipeline, "features_needed") else None
    per_run = {}
    for run in runs:
        sample = load_sample(run, features=needed)
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
