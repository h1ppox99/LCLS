"""Pure metrics for comparing evidence masks and mask ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class MaskDelta:
    """Difference from a baseline mask over one fixed evaluation domain."""

    iou: float
    changed_fraction: float
    added_fraction: float
    removed_fraction: float

    @property
    def volume_delta(self) -> float:
        return self.added_fraction - self.removed_fraction

    def as_dict(self) -> dict:
        return {
            "iou": self.iou,
            "changed_fraction": self.changed_fraction,
            "added_fraction": self.added_fraction,
            "removed_fraction": self.removed_fraction,
            "volume_delta": self.volume_delta,
        }


def _boolean(name: str, value, shape=None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.bool_:
        raise TypeError(f"{name} must be boolean")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != {shape}")
    return arr


def compare_masks(candidate, baseline, domain) -> MaskDelta:
    """Compare two masks after restricting both to ``domain``."""
    domain = _boolean("domain", domain)
    candidate = _boolean("candidate", candidate, domain.shape) & domain
    baseline = _boolean("baseline", baseline, domain.shape) & domain
    size = int(domain.sum())
    if not size:
        raise ValueError("mask comparison domain is empty")

    union = int((candidate | baseline).sum())
    intersection = int((candidate & baseline).sum())
    added = int((candidate & ~baseline).sum())
    removed = int((baseline & ~candidate).sum())
    return MaskDelta(
        iou=float(intersection) / union if union else 1.0,
        changed_fraction=float(added + removed) / size,
        added_fraction=float(added) / size,
        removed_fraction=float(removed) / size,
    )


def selection_frequency(masks: Iterable[np.ndarray]) -> np.ndarray:
    """Fraction of an explicitly declared mask ensemble selecting each pixel."""
    masks = tuple(masks)
    if not masks:
        raise ValueError("selection frequency needs at least one mask")
    first = _boolean("mask", masks[0])
    count = np.zeros(first.shape, dtype=np.uint32)
    for mask in masks:
        count += _boolean("mask", mask, first.shape)
    return count.astype(np.float64) / len(masks)


def instability(masks: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return selection frequency and pairwise disagreement probability."""
    frequency = selection_frequency(masks)
    return frequency, 2.0 * frequency * (1.0 - frequency)
