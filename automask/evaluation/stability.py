"""Empirical stability diagnostics built from disjoint shot folds."""
from __future__ import annotations

import itertools

import numpy as np

from automask.evaluation.resampling import FoldMoments, fold_sample


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def sampling_stability(
    pipeline,
    sample,
    moments: FoldMoments,
    rng: np.random.Generator,
    repetitions: int = 20,
) -> np.ndarray:
    """Mask IoU across repeated complementary halves of dealt shot folds."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    half = moments.k // 2
    if half == 0 or moments.k % 2:
        raise ValueError("sampling stability requires an even number of folds")

    folds = np.arange(moments.k)
    partitions = [np.asarray(part) for part in itertools.combinations(folds, half)
                  if 0 in part]
    if repetitions > len(partitions):
        raise ValueError(
            f"{moments.k} folds provide only {len(partitions)} unique half-splits"
        )
    chosen = rng.choice(len(partitions), size=repetitions, replace=False)
    values = np.empty(repetitions, dtype=np.float64)
    for i, index in enumerate(chosen):
        left = partitions[int(index)]
        right = np.setdiff1d(folds, left)
        a = pipeline.run(fold_sample(sample, moments, left, dealt=True))
        b = pipeline.run(fold_sample(sample, moments, right, dealt=True))
        values[i] = mask_iou(a, b)
    return values


def temporal_stability(pipeline, sample, moments: FoldMoments) -> float:
    """Mask IoU between the chronological first and second halves of a run."""
    early, late = moments.halves()
    a = pipeline.run(fold_sample(sample, moments, early, dealt=False))
    b = pipeline.run(fold_sample(sample, moments, late, dealt=False))
    return mask_iou(a, b)
