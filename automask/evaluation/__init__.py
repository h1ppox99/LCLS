"""Labelled evaluation and ground-truth-free consistency diagnostics."""

from automask.evaluation.consistency import evaluate_consistency
from automask.evaluation.labelled import (
    ALL_RUNS,
    FIT_RUNS,
    VALIDATION_RUNS,
    evaluate,
    reference_mask,
)
from automask.evaluation.metrics import MaskDelta, compare_masks
from automask.evaluation.report import MaskValidationReport
from automask.evaluation.validation import (
    MaskValidationDesign,
    ParameterSweep,
    validate_mask,
)

__all__ = [
    "ALL_RUNS",
    "FIT_RUNS",
    "VALIDATION_RUNS",
    "evaluate",
    "evaluate_consistency",
    "MaskDelta",
    "MaskValidationDesign",
    "MaskValidationReport",
    "ParameterSweep",
    "compare_masks",
    "reference_mask",
    "validate_mask",
]
