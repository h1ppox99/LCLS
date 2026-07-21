"""
stats/calib.py -- psana calibration bad-pixel floor statistic.

The `pixel_status` mask frozen per run into data/masks/statusMask_run<NNNN>_asm.npy,
loaded onto the Sample by evaluation.load_sample. FLOOR stat: boolean, no sweep.
"""
from __future__ import annotations
from dataclasses import dataclass

from stats.base import StatSpec, register_stat


@dataclass
class CalibParams:
    pass


def compute(sample, params: CalibParams | None = None):
    return sample.calib


register_stat(StatSpec(
    name="calib",
    compute=compute,
    params=CalibParams,
    kind="floor",
    needs=("calib",),
    doc="psana pixel_status bad-pixel mask; 100%-precision floor",
))
