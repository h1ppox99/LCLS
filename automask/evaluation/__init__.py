"""Labelled evaluation and ground-truth-free consistency diagnostics."""
from automask.evaluation.consistency import evaluate_consistency
from automask.evaluation.labelled import (
    ALL_RUNS,
    FIT_RUNS,
    VALIDATION_RUNS,
    evaluate,
    reference_mask,
)

__all__ = [
    "ALL_RUNS",
    "FIT_RUNS",
    "VALIDATION_RUNS",
    "evaluate",
    "evaluate_consistency",
    "reference_mask",
]
