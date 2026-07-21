"""
stats/base.py -- registry + shared low-level helpers for statistic modules.

A "statistic" turns a per-run :class:`~evaluation.Sample` into a continuous
per-pixel evidence field (or, for the intensity-free FLOOR statistics, straight
into a boolean mask). Every stat module registers one :class:`StatSpec` here so
masking.py and the sweep driver can look them up by name.

Two kinds:
  * kind="field"  -- compute(sample, params) -> float field. By convention the
    sign follows the RAW detector (e.g. variance flags LOW z); `mode` on the spec
    says which side is defect-like so the pipeline can threshold and sign-fold it.
  * kind="floor"  -- compute(sample, params) -> bool mask (True == masked). These
    are the 100%-precision geometry/calibration masks; they carry no sweep space.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Type

import numpy as np

# name -> StatSpec. Populated at import time by each stat module via register_stat.
STATS: Dict[str, "StatSpec"] = {}


@dataclass
class StatSpec:
    name: str
    compute: Callable            # (sample, params) -> np.ndarray (field) or bool mask (floor)
    params: Type                 # dataclass type holding this stat's hyperparameters
    kind: str = "field"          # "field" or "floor"
    mode: str = "low"            # default defect side for field stats: low|high|both
    needs: Tuple[str, ...] = ()  # Sample attributes this stat reads (documentation)
    doc: str = ""

    @property
    def swept(self) -> bool:
        """Floor stats are always-on and carry no hyperparameters to sweep."""
        return self.kind != "floor"


def register_stat(spec: StatSpec) -> StatSpec:
    STATS[spec.name] = spec
    return spec


# --------------------------------------------------------------------------
#  shared numerics
# --------------------------------------------------------------------------
def robust_z(field_arr, domain, transform=None):
    """Signed robust-MAD z-score of `field_arr` over boolean `domain`, 0 outside
    it. `transform` (if given) is applied AFTER indexing into `domain`."""
    v = field_arr[domain]
    if transform is not None:
        v = transform(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    z = np.zeros(field_arr.shape, dtype=np.float64)
    z[domain] = (v - med) / (mad + 1e-12)
    return z


def threshold_stat(z, k, mode="low"):
    """Threshold a continuous field: "low" -> z < -k, "high" -> z > k, "both" ->
    either. Callers gate by `real`/domain themselves."""
    hi, lo = (z > k), (z < -k)
    return {"low": lo, "high": hi, "both": hi | lo}[mode]
