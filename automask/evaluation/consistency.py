"""Mask reproducibility across fixed real-shot splits."""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from automask.image_store import REDUCTIONS, ImageStore
from automask.sample import DERIVED, Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection


def _iou(a, b) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def _evidence_mask(pipeline, sample, shape):
    mask = np.asarray(pipeline.run(sample))
    floor = np.asarray(pipeline.floor(sample))
    if mask.dtype != np.bool_ or floor.dtype != np.bool_:
        raise TypeError("a pipeline mask and floor must be boolean")
    if mask.shape != shape or floor.shape != shape:
        raise TypeError("a pipeline must return a boolean mask with the floor shape")
    return mask & ~floor


def evaluate_consistency(
    pipeline,
    run: int,
    selection: ShotSelection = BEAM_ON_SELECTION,
    store: Optional[ImageStore] = None,
    n_folds: int = 10,
) -> dict:
    """Compare masks from round-robin folds and two chronological halves."""
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError(f"n_folds must be an integer >= 2, got {n_folds!r}")
    store = store or ImageStore()
    reductions = sorted(({"mean", *pipeline.needs()} - set(DERIVED)) & REDUCTIONS)
    folds = {name: store.folds(run, selection, name, n_folds=n_folds)
             for name in reductions}
    halves = {name: store.halves(run, selection, name, n_folds=n_folds)
              for name in reductions}

    sample = Sample.from_store(run, selection, pipeline.needs(), store=store)
    floor = np.asarray(pipeline.floor(sample))
    if floor.dtype != np.bool_:
        raise TypeError("a pipeline floor must be boolean")
    full = _evidence_mask(pipeline, sample, floor.shape)

    fold_masks = [
        _evidence_mask(pipeline, sample.with_arrays(**{
            name: folds[name][i] for name in reductions
        }), floor.shape)
        for i in range(n_folds)
    ]
    half_masks = [
        _evidence_mask(pipeline, sample.with_arrays(**{
            name: halves[name][i] for name in reductions
        }), floor.shape)
        for i in range(2)
    ]
    pairwise = np.array([_iou(fold_masks[i], fold_masks[j])
                         for i, j in itertools.combinations(range(n_folds), 2)])
    versus_full = np.array([_iou(mask, full) for mask in fold_masks])
    return {
        "n_folds": n_folds,
        "fold_iou": pairwise,
        "fold_iou_mean": float(pairwise.mean()),
        "fold_iou_std": float(pairwise.std()),
        "fold_vs_full_iou": versus_full,
        "fold_vs_full_iou_mean": float(versus_full.mean()),
        "fold_vs_full_iou_std": float(versus_full.std()),
        "chronological_iou": _iou(*half_masks),
    }
