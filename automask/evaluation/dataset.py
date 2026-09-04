"""
dataset.py -- numpy-only access to the frozen REFERENCE masks.

No psana, no h5py, no LCLS filesystem. Only the hand-drawn masks live here: they
are measurements someone made once, so they cannot be recomputed. Every array a
pipeline consumes comes from `ImageStore` instead (see `automask.sample`).
All masks follow one convention: bool, True == masked.

    from automask.evaluation.dataset import reference_mask_path, reference_runs, score

    gt = np.load(reference_mask_path(475))  # when a run's reference is packaged
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_MSK = Path(__file__).resolve().parents[1] / "reference_masks"
_PATTERN = re.compile(r"^reference_mask_run(\d+)$")


def reference_mask_path(run: int) -> Path:
    """Path to the run's packaged reference mask (may not exist)."""
    return _MSK / f"reference_mask_run{run}.npy"


def list_masks() -> list[str]:
    return sorted(path.stem for path in _MSK.glob("*.npy"))


def reference_runs() -> tuple[int, ...]:
    """Runs with a packaged run-specific hand mask."""
    runs = {
        int(match.group(1)) for name in list_masks() if (match := _PATTERN.match(name))
    }
    return tuple(sorted(runs))


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """IoU / precision / recall of a predicted mask vs a reference (True==masked)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"iou": iou, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    print("masks:", list_masks())
