"""
features/base.py -- a feature is ``reduction ∘ ShotSelection``.

A masking statistic doesn't consume a raw detector; it consumes a *feature*: a
per-pixel reduction (mean/std/median/mad) over a set of shots chosen by a
:class:`~automask.shot_selection.ShotSelection`. This module makes that binding a
first-class object so the dependency chain is explicit:

    Pipeline -> Stat.needs (feature names) -> FeatureSpec -> ShotSelection -> XTC

``FEATURES`` is the catalogue (name -> FeatureSpec), mirroring the STATS /
REGULARIZERS / COMBINERS registries in the masking layer. It is populated by
importing ``automask.features.catalog``. Resolving a spec to an array (compute
from XTC + cache) is the job of :class:`~automask.features.store.FeatureStore`;
this module stays psana-free and import-cheap.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Literal, Optional

from automask.shot_selection import ShotSelection

# mean/std stream in one pass; median/mad stage frames to disk first (see store.py).
Reduction = Literal["mean", "std", "median", "mad"]

Source = Literal["events", "calib"]

Form = Literal["asm", "panel"]


@dataclass(frozen=True)
class FeatureSpec:
    """One feature: either a ``reduction`` over the frames a ``selection`` keeps
    (``source="events"``), or a psana calibration ``constant`` at one gain stage
    (``source="calib"``).

    ``name`` is the catalogue label a stat's ``needs`` refers to (e.g. ``umean``);
    identity for caching is everything that determines the numbers, name excluded
    -- so two names with the same reduction and selection share one cached array.
    """

    name: str
    reduction: Optional[Reduction] = None
    selection: Optional[ShotSelection] = None
    source: Source = "events"
    form: Form = "asm"
    constant: Optional[str] = None   # calib: psana constant, e.g. "pedestals"
    gain: Optional[int] = None       # calib: gain stage index (0 == high gain)

    def __post_init__(self) -> None:
        if self.source == "events":
            if self.reduction is None or self.selection is None:
                raise ValueError(
                    f"feature {self.name!r}: source='events' needs both a "
                    f"reduction and a selection")
        elif self.source == "calib":
            if self.constant is None or self.gain is None:
                raise ValueError(
                    f"feature {self.name!r}: source='calib' needs both a "
                    f"constant name and a gain stage")
        else:
            raise ValueError(f"feature {self.name!r}: unknown source {self.source!r}")

    @property
    def content_key(self) -> str:
        """Provenance hash of what actually determines the array (name excluded).

        The ``source="events"`` payload is the reduction plus ``asdict`` of the
        selection, so any change to ``ShotSelection``'s field names invalidates
        every cached entry. 
        """
        if self.source == "events":
            payload = {"reduction": self.reduction,
                       "selection": asdict(self.selection)}
        else:
            payload = {"source": self.source, "constant": self.constant,
                       "gain": self.gain, "form": self.form}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def cache_stub(self, run: int) -> str:
        """Filename stem for this feature's cache entry (form suffix added later)."""
        stem = self.reduction if self.source == "events" else \
            f"{self.constant}g{self.gain}"
        return f"{stem}_{self.content_key}_run{run:04d}"


FEATURES: Dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    FEATURES[spec.name] = spec
    return spec


def get_spec(name: str) -> FeatureSpec:
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {sorted(FEATURES)}")
    return FEATURES[name]
