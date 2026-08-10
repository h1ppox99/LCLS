"""
shot_selection.py -- decide *which* XTC shots to build detector features from.

Every scalar a selection needs (beam state, CC/VCC branch state, per-shot
intensity monitors) is taken from the canonical columns produced by
``automask.utils.profile_run_values``.

Two objects:

* ``ShotMeta``      -- the selection-specific view of a canonical run profile.
                       Pure data, no psana.
* ``ShotSelection`` -- a declarative spec (beam class, CC/VCC branch classes,
                       shot count, intensity percentile trim, normalization) plus
                       a pure-numpy ``resolve(meta) -> event indices``. No psana
                       here either, so the selection logic is unit-testable.

Provider/consumer split: the profiler reads psana and official
``smalldata_tools`` adapters; this module only interprets its numpy columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

BeamClass = Literal["on", "off", "any"]
BranchClass = Literal["open", "closed", "any"]

#: Per-shot intensity monitors, in rough order of usefulness for this experiment.
#: The first three are downstream of the CC/VCC split and track what the detector
#: actually receives; ipm2 and gasdet are upstream and do not.
MONITORS = ("sample_diode", "diodeU", "lombpm", "ipm2", "gasdet")

#: Value of ``ShotSelection.normalization`` meaning "do not rescale frames".
NO_NORMALIZATION = "none"

CC_VCC_THRESHOLD = 2.0

_SHOT_META_COLUMNS = {
    "beam_on": "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]",
    "cc_voltage": "ai/ch02",
    "vcc_voltage": "ai/ch03",
}

_INTENSITY_COLUMNS = {
    "sample_diode": "diodeU/channels[0]",
    "diodeU": "diodeU/sum",
    "lombpm": "lombpm/sum",
    "ipm2": "ipm2/sum",
    "gasdet": "gas_detector/f_11_ENRC",
}


@dataclass
class ShotMeta:
    """Per-shot scalar table for one run, one row per event in stream order.

    Built from ``profile_run_values(...)["values"]``; no detector frames are
    decoded by that profiler pass.

    ``beam_on`` is EVR code 137. ``cc_open``/``vcc_open`` are the CC/VCC shutter
    voltages thresholded at ``CC_VCC_THRESHOLD``. ``intensity`` maps a
    monitor name in ``MONITORS`` to its per-shot readings; a monitor absent from
    the run may be reported as all-NaN rather than omitted.
    """

    run: int
    beam_on: np.ndarray
    cc_open: np.ndarray
    vcc_open: np.ndarray
    intensity: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: dict) -> "ShotMeta":
        """Interpret the selection fields in a canonical run profile."""
        columns = profile.get("values", {})
        missing = [
            field for field in _SHOT_META_COLUMNS.values() if field not in columns
        ]
        if missing:
            raise KeyError(
                f"run profile is missing required shot-selection columns: {missing}"
            )

        intensity = {
            monitor: np.asarray(columns[field], dtype=np.float64)
            for monitor, field in _INTENSITY_COLUMNS.items()
            if field in columns
        }
        if not intensity:
            raise KeyError(
                "run profile has none of the supported intensity columns: "
                f"{list(_INTENSITY_COLUMNS.values())}"
            )

        beam = np.asarray(columns[_SHOT_META_COLUMNS["beam_on"]], dtype=float)
        cc = np.asarray(columns[_SHOT_META_COLUMNS["cc_voltage"]], dtype=float)
        vcc = np.asarray(columns[_SHOT_META_COLUMNS["vcc_voltage"]], dtype=float)
        return cls(
            run=int(profile["run"]),
            beam_on=np.isfinite(beam) & (beam > 0.5),
            cc_open=np.isfinite(cc) & (cc > CC_VCC_THRESHOLD),
            vcc_open=np.isfinite(vcc) & (vcc > CC_VCC_THRESHOLD),
            intensity=intensity,
        )

    def __post_init__(self) -> None:
        self.beam_on = np.asarray(self.beam_on, dtype=bool)
        self.cc_open = np.asarray(self.cc_open, dtype=bool)
        self.vcc_open = np.asarray(self.vcc_open, dtype=bool)
        n = self.beam_on.shape[0]
        if not (self.cc_open.shape[0] == self.vcc_open.shape[0] == n):
            raise ValueError("ShotMeta beam/cc/vcc arrays must share length")
        if not self.intensity:
            raise ValueError("ShotMeta needs at least one intensity monitor")
        self.intensity = {k: np.asarray(v, dtype=np.float64)
                          for k, v in self.intensity.items()}
        for name, values in self.intensity.items():
            if name not in MONITORS:
                raise ValueError(
                    f"unknown monitor {name!r}; known: {list(MONITORS)}")
            if values.shape[0] != n:
                raise ValueError(
                    f"monitor {name!r} has {values.shape[0]} rows, expected {n}")

    @property
    def n_events(self) -> int:
        return self.beam_on.shape[0]

    def monitor(self, name: str) -> np.ndarray:
        """Readings for one monitor, with a clear error if the scan lacked it."""
        if name not in self.intensity:
            raise KeyError(
                f"run {self.run}: monitor {name!r} was not scanned; "
                f"available: {sorted(self.intensity)}")
        return self.intensity[name]


@dataclass(frozen=True)
class ShotSelection:
    """Declarative recipe for which shots feed a feature build, plus how their
    frames are combined.

    Knobs:

    * ``beam``   -- keep shots where the machine delivered x-rays (EVR 137):
                    ``"on"`` (lit-beam features: umean/ustd), ``"off"`` (beam-off
                    dark: mean), or ``"any"``.
    * ``cc``     -- CC branch shutter: ``"open"`` / ``"closed"`` / ``"any"``.
    * ``vcc``    -- VCC branch shutter, same classes. Independent of ``cc`` and
                    ANDed with it, so all four branch states are expressible.
    * ``n_shots``-- how many shots to keep (``None`` = every survivor).
    * ``filter_low``  -- drop the lowest fraction of survivors by intensity.
    * ``filter_high`` -- drop the highest fraction of survivors by intensity.
    * ``intensity``   -- which monitor in ``MONITORS`` defines "intensity" for
                    the validity floor and the percentile trim.
    * ``normalization`` -- how the *feature builder* combines the selected frames.
                    ``"none"``: accumulate the psana-calibrated frames as-is.
                    A monitor name: additionally scale each frame by
                    ``median(i0) / i0[shot]`` before accumulating.
                    NOTE: this is orthogonal to calibration -- frames are
                    ALWAYS psana-calibrated (pedestal+gain+common-mode);
                    ``normalization`` only controls the optional per-shot i0
                    scaling layered on top. Meant for ``beam="on"`` builds.
    """

    beam: BeamClass = "on"
    cc: BranchClass = "open"
    vcc: BranchClass = "any"
    n_shots: Optional[int] = 800
    filter_low: float = 0.03
    filter_high: float = 0.03
    intensity: str = "sample_diode"
    normalization: str = NO_NORMALIZATION

    def __post_init__(self) -> None:
        if not (0.0 <= self.filter_low < 1.0) or not (0.0 <= self.filter_high < 1.0):
            raise ValueError("filter_low/filter_high must be in [0, 1)")
        if self.filter_low + self.filter_high >= 1.0:
            raise ValueError("filter_low + filter_high must leave some shots")
        if self.n_shots is not None and self.n_shots < 1:
            raise ValueError("n_shots must be positive or None")
        if self.intensity not in MONITORS:
            raise ValueError(
                f"unknown intensity monitor {self.intensity!r}; "
                f"known: {list(MONITORS)}")
        if self.normalization != NO_NORMALIZATION and self.normalization not in MONITORS:
            raise ValueError(
                f"unknown normalization monitor {self.normalization!r}; "
                f"use {NO_NORMALIZATION!r} or one of {list(MONITORS)}")

    @staticmethod
    def _branch_mask(state: np.ndarray, want: BranchClass) -> np.ndarray:
        if want == "open":
            return state
        if want == "closed":
            return ~state
        return np.ones_like(state, dtype=bool)

    def _valid(self, meta: ShotMeta) -> np.ndarray:
        """Shots whose intensity reading is usable: finite and non-zero.

        Mirrors the lab's own gate (``xpp_sharing/utils.py``: reject non-finite
        or zero sample diode). Unlike the previous ``> 0`` floor this is applied
        for every beam class -- a zero reading is unusable as a trim axis and as
        a normalizer whether or not the beam was on.
        """
        values = meta.monitor(self.intensity)
        return np.isfinite(values) & (values != 0)

    def _shot_class(self, meta: ShotMeta) -> np.ndarray:
        """Boolean mask of shots matching the beam AND cc AND vcc classes, with
        the intensity validity floor applied."""
        keep = self._valid(meta)
        if self.beam == "on":
            keep &= meta.beam_on
        elif self.beam == "off":
            keep &= ~meta.beam_on
        keep &= self._branch_mask(meta.cc_open, self.cc)
        keep &= self._branch_mask(meta.vcc_open, self.vcc)
        return keep

    def resolve(self, meta: ShotMeta) -> np.ndarray:
        """Return the event indices (into ``meta``, stream order) to build from.

        valid beam/cc/vcc class -> drop intensity below ``q(filter_low)`` / above
        ``q(1 - filter_high)`` -> take ``n_shots`` evenly spaced across the
        survivors.
        """
        in_class = self._shot_class(meta)
        values = meta.monitor(self.intensity)
        vals = values[in_class]
        if vals.size == 0:
            raise RuntimeError(
                f"run {meta.run}: no shots match beam={self.beam!r}, "
                f"cc={self.cc!r}, vcc={self.vcc!r} on monitor "
                f"{self.intensity!r} (of {meta.n_events} events); "
                f"breakdown: {self.describe(meta)}")
        lo = np.quantile(vals, self.filter_low)
        hi = np.quantile(vals, 1.0 - self.filter_high)
        keep = in_class & (values >= lo) & (values <= hi)
        survivors = np.flatnonzero(keep)
        if self.n_shots is None or self.n_shots >= survivors.size:
            return survivors
        # TODO: Refine selection strategy later
        pick = np.linspace(0, survivors.size - 1, self.n_shots).round().astype(np.int64)
        return survivors[np.unique(pick)]

    @staticmethod
    def create_k_folds(indices: np.ndarray, k: int = 1) -> list:
        """Split resolved event indices into ``k`` interchangeable folds.

        Shots are dealt round-robin, so consecutive *selected* shots land in
        different folds. 

        ``k=1`` returns ``[indices]`` unchanged.
        """
        indices = np.asarray(indices)
        if k < 1 or k > indices.size:
            raise ValueError(f"k must be in [1, {indices.size}], got {k}")
        return [indices[f::k] for f in range(k)]

    def describe(self, meta: ShotMeta) -> dict:
        """Per-filter shot-count breakdown over ``meta``: how many events survive
        each filter alone, and how many are *accessible* under all of them
        (before the intensity trim / ``n_shots`` cap). Pure, testable, no psana.
        """
        beam = (meta.beam_on if self.beam == "on" else
                ~meta.beam_on if self.beam == "off" else
                np.ones(meta.n_events, dtype=bool))
        return {"n_events": int(meta.n_events),
                "n_valid": int(self._valid(meta).sum()),
                "n_beam": int(beam.sum()),
                "n_cc": int(self._branch_mask(meta.cc_open, self.cc).sum()),
                "n_vcc": int(self._branch_mask(meta.vcc_open, self.vcc).sum()),
                "n_accessible": int(self._shot_class(meta).sum())}

    def reference_intensity(self, meta: ShotMeta, indices: np.ndarray) -> float:
        """Reference i0 for a non-``"none"`` ``normalization``: median reading of
        the normalization monitor over the selected shots. Frames are then scaled
        by ``reference / i0[shot]``."""
        return float(np.median(meta.monitor(self.normalization)[indices]))
