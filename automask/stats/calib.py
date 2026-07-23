"""Calibration-status bad-pixel floor statistic."""
from __future__ import annotations

from dataclasses import dataclass

from automask.stats.base import StatSpec, register_stat


@dataclass
class CalibParams:
    """No parameters: the per-run status mask is supplied by ``Sample``."""


def compute(sample, params: CalibParams | None = None):
    """Return the frozen psana pixel-status mask (``True == masked``)."""
    return sample.calib


register_stat(StatSpec(
    name="calib",
    compute=compute,
    params=CalibParams,
    kind="floor",
    needs=("calib",),
    doc="psana pixel-status bad-pixel mask; intensity-free floor",
))
