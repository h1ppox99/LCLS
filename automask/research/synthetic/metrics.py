"""
synthetic/metrics.py -- region-restricted detection metrics.

All scoring happens over the EVALUATION REGION only: the pixels that were valid
in the original ground-truth mask (``region = ~ground_truth``). Restricting to
this region is deliberate -- it stops us penalising a masker for (correctly)
flagging real pre-existing artifacts that were never part of the injection.

Every ratio has an explicit zero-denominator fallback. The precision / recall /
IoU conventions match ``automask.evaluation.dataset.score`` (empty-vs-empty -> 1.0).
"""

from __future__ import annotations

import numpy as np


def _ratio(num, den, default: float) -> float:
    """num/den, or `default` when the denominator is zero."""
    return float(num) / float(den) if den else float(default)


def masking_metrics(pred: np.ndarray, injected: np.ndarray, region: np.ndarray) -> dict:
    """Score ``pred`` against ``injected``, both restricted to ``region``.

    Parameters are boolean arrays of identical shape (True == masked / present).
    ``region`` is the originally-valid mask. Returns precision, recall, f1, iou,
    fpr (false-positive rate over region negatives) and masked_frac (fraction of
    region pixels the prediction masks), plus the raw tp/fp/fn/tn counts.
    """
    pred = pred.astype(bool) & region
    truth = injected.astype(bool) & region
    neg = region & ~truth  # originally-valid non-artifact px

    tp = int((pred & truth).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & neg).sum())

    precision = _ratio(tp, tp + fp, 1.0)  # no positives predicted -> 1.0
    recall = _ratio(tp, tp + fn, 1.0)  # nothing to find -> 1.0
    iou = _ratio(tp, tp + fp + fn, 1.0)
    f1 = _ratio(2 * precision * recall, precision + recall, 0.0)
    fpr = _ratio(fp, fp + tn, 0.0)  # no negatives -> 0.0
    masked_frac = _ratio(int(pred.sum()), int(region.sum()), 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr,
        "masked_frac": masked_frac,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
