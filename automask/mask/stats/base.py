"""
stats/base.py -- registry + shared low-level helpers for statistic modules.

A "statistic" turns a per-run :class:`~evaluation.Sample` into a continuous
per-pixel evidence field (or, for the intensity-free FLOOR statistics, straight
into a boolean mask). Every stat module registers one :class:`StatSpec` here so
masking.py can look them up by name.

Three kinds:
  * kind="field"  -- compute(sample, params) -> float field. By convention the
    sign follows the RAW detector (e.g. variance flags LOW z); `mode` on the spec
    says which side is defect-like so the pipeline can threshold it. A robust-z
    scale is the convention so a single `k` reads the same across stats.
  * kind="pick"   -- compute(sample, params) -> bool mask (True == masked), for
    detectors whose decision is not per-pixel and so has no meaningful graded
    field: `hough_lines` votes on SEGMENTS, and a pixel is on one or it is not.
    Its knobs live upstream of any threshold, so it carries no `k`/`mode` and
    skips the field-regularizer + threshold stages of Detector entirely (both
    are errors on a pick -- see Detector.field). Unlike a floor, a pick has
    tunable parameters and is not 100%-precision.
  * kind="floor"  -- compute(sample, params) -> bool mask (True == masked). These
    are the 100%-precision geometry/calibration masks; they carry no parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np

# name -> StatSpec. Populated at import time by each stat module via register_stat.
STATS: Dict[str, "StatSpec"] = {}


@dataclass
class Panel:
    """One diagnostic image for the explain views (viz.explain_panels).

    ``mask=False`` is a graded field, shown as an image plus its distribution;
    ``threshold`` (k, mode), if set, is marked on that distribution. ``mask=True``
    is a boolean decision, shown on its own. This is what a stat's ``explain``
    hook returns per panel -- the graded evidence a pick thresholds internally is
    a `Panel`, so a pick gets the same field-vs-cut view a field channel does.
    """

    array: np.ndarray
    threshold: Optional[Tuple[float, str]] = None
    mask: bool = False


@dataclass
class StatSpec:
    name: str
    compute: Callable  # (sample, params) -> float field, or bool mask (pick/floor)
    params: Type  # dataclass type holding this stat's hyperparameters
    kind: str = "field"  # "field", "pick" or "floor"
    mode: str = "low"  # default defect side for field stats: low|high|both
    needs: Tuple[str, ...] = ()  # Sample attributes this stat reads (documentation)
    doc: str = ""
    # (sample, params) -> {name: Panel}: the stat's internal evidence, for a stat
    # whose one output does not tell the whole story (a pick's darkness field,
    # binary input and segments). A field stat needs none -- its `field()` IS the
    # evidence -- so Channel.explain builds the default from it.
    explain: Optional[Callable] = None

    @property
    def swept(self) -> bool:
        """Floor stats are always-on and carry no hyperparameters to sweep."""
        return self.kind != "floor"

    @property
    def emits_mask(self) -> bool:
        """True when `compute` returns a boolean mask rather than a float field."""
        return self.kind in ("pick", "floor")


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
