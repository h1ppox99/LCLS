"""Typed outputs shared by evaluation entry points."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Estimate:
    value: float
    standard_deviation: Optional[float]
    n: int


@dataclass(frozen=True)
class RuntimeEvaluation:
    run: int
    masked_fraction: float
    floor_contained: bool
    sampling_stability: Estimate
    temporal_stability: Estimate
    azimuthal_excess: Estimate
    azimuthal_gain: Estimate
    azimuthal_win_rate: Estimate

    def as_dict(self) -> dict:
        return asdict(self)
