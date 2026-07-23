"""
shot_selection.py -- decide *which* XTC shots to build detector features from.

This is the grounded replacement for the ad-hoc `_selection` helpers that used
to live inside individual producers and lean on the imported small-data file.
Every scalar a selection needs (per-shot intensity, x-ray on/off) is available
directly from the XTC stream via psana, so features can be rebuilt from raw data
with no small-data dependency (see `automask.io.read_xtc.scan_shots`).

Two objects:

* ``ShotMeta``      -- the per-shot scalar table produced by one cheap pass over
                       the run (intensity + x-ray/laser flags). Pure data, no psana.
* ``ShotSelection`` -- a declarative spec (x-ray class, shot count, intensity
                       percentile trim, normalization) plus a pure-numpy
                       ``resolve(meta) -> event indices``. No psana here either,
                       so the selection logic is trivially unit-testable.

Provider/consumer split: the only psana-touching code is the scan that fills a
``ShotMeta`` (in ``automask.io.read_xtc``). Porting to the SLAC cluster means
swapping that one scan from explicit local stream files to
``psana.DataSource('exp=xppl1016922:run=N:smd')`` -- everything in this module is
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

XRayClass = Literal["on", "off", "any"]
LaserClass = Literal["on", "off", "any"]
Normalization = Literal["none", "ipm2"]


@dataclass
class ShotMeta:
    """Per-shot scalar table for one run, one row per event in stream order.

    Built by a single cheap pass over the XTC (scalars only, no Jungfrau
    ``.calib()``); see ``automask.io.read_xtc.scan_shots``. ``intensity`` is the
    ipm2 BMMON total intensity; ``xray_on``/``laser_on`` come from EVR codes
    (x-ray requires code 137, laser is dropped on code 91 -- both conventions
    read from the run's ``UserDataCfg``).
    """

    run: int
    intensity: np.ndarray  # (N,) float, ipm2 TotalIntensity per shot
    xray_on: np.ndarray    # (N,) bool
    laser_on: np.ndarray   # (N,) bool

    def __post_init__(self) -> None:
        self.intensity = np.asarray(self.intensity, dtype=np.float64)
        self.xray_on = np.asarray(self.xray_on, dtype=bool)
        self.laser_on = np.asarray(self.laser_on, dtype=bool)
        n = self.intensity.shape[0]
        if not (self.xray_on.shape[0] == self.laser_on.shape[0] == n):
            raise ValueError("ShotMeta arrays must share length")

    @property
    def n_events(self) -> int:
        return self.intensity.shape[0]


@dataclass(frozen=True)
class ShotSelection:
    """Declarative recipe for which shots feed a feature build, plus how their
    frames are combined.

    Baseline knobs:

    * ``xray``       -- keep x-ray ``"on"`` shots (lit-beam features: umean/ustd),
                        ``"off"`` shots (beam-off dark: mean), or ``"any"``.
    * ``laser``      -- keep laser ``"on"`` / ``"off"`` / ``"any"`` shots
                        (pump-probe pump state). Combined with ``xray`` (AND).
    * ``n_shots``    -- how many shots to keep (``None`` = every survivor).
    * ``filter_low`` -- drop the lowest fraction of survivors by intensity (3%).
    * ``filter_high``-- drop the highest fraction of survivors by intensity (3%).
    * ``normalization`` -- how the *feature builder* combines the selected frames.
                        ``"none"``: accumulate the psana-calibrated frames as-is.
                        ``"ipm2"``: additionally scale each frame by
                        ``median(i0) / i0[shot]`` before accumulating.
                        NOTE: this is orthogonal to calibration -- frames are
                        ALWAYS psana-calibrated (pedestal+gain+common-mode);
                        ``normalization`` only controls the optional per-shot i0
                        scaling layered on top. Meant for ``xray="on"`` builds.

    ``resolve`` is pure numpy and consumes only a ``ShotMeta`` -- no psana, no
    small-data. ``normalization`` is carried here (not used by ``resolve``) so a
    single object fully specifies one feature extraction.
    """

    xray: XRayClass = "on"
    laser: LaserClass = "any"
    n_shots: Optional[int] = None
    filter_low: float = 0.03
    filter_high: float = 0.03
    normalization: Normalization = "none"

    def __post_init__(self) -> None:
        if not (0.0 <= self.filter_low < 1.0) or not (0.0 <= self.filter_high < 1.0):
            raise ValueError("filter_low/filter_high must be in [0, 1)")
        if self.filter_low + self.filter_high >= 1.0:
            raise ValueError("filter_low + filter_high must leave some shots")
        if self.n_shots is not None and self.n_shots < 1:
            raise ValueError("n_shots must be positive or None")

    def _shot_class(self, meta: ShotMeta) -> np.ndarray:
        """Boolean mask of shots in the requested x-ray AND laser class, with a
        validity floor. x-ray-on/any require a positive i0 (physical beam, and
        safe for ipm2 normalization); x-ray-off only requires a finite reading,
        since beam-off intensity sits at noise around zero. The laser class is an
        independent AND filter on the pump state."""
        finite = np.isfinite(meta.intensity)
        if self.xray == "on":
            keep = finite & (meta.intensity > 0) & meta.xray_on
        elif self.xray == "off":
            keep = finite & ~meta.xray_on
        else:                                 # "any"
            keep = finite & (meta.intensity > 0)
        if self.laser == "on":
            keep &= meta.laser_on
        elif self.laser == "off":
            keep &= ~meta.laser_on
        return keep

    def resolve(self, meta: ShotMeta) -> np.ndarray:
        """Return the event indices (into ``meta``, stream order) to build from.

        valid x-ray/laser class -> drop intensity below ``q(filter_low)`` / above
        ``q(1 - filter_high)`` -> take ``n_shots`` evenly spaced across the
        survivors.
        """
        in_class = self._shot_class(meta)
        vals = meta.intensity[in_class]
        if vals.size == 0:
            raise RuntimeError(
                f"run {meta.run}: no shots match xray={self.xray!r}, "
                f"laser={self.laser!r} (of {meta.n_events} events)"
            )
        lo = np.quantile(vals, self.filter_low)
        hi = np.quantile(vals, 1.0 - self.filter_high)
        keep = in_class & (meta.intensity >= lo) & (meta.intensity <= hi)
        survivors = np.flatnonzero(keep)
        if self.n_shots is None or self.n_shots >= survivors.size:
            return survivors
        # TODO: Refine selection strategy later
        pick = np.linspace(0, survivors.size - 1, self.n_shots).round().astype(np.int64)  # TODO: Refine selection strategy later
        return survivors[np.unique(pick)]

    def reference_intensity(self, meta: ShotMeta, indices: np.ndarray) -> float:
        """Reference i0 for ``normalization='ipm2'``: median intensity over the
        selected shots. Frames are then scaled by ``reference / i0[shot]``."""
        return float(np.median(meta.intensity[indices]))
