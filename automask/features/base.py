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
from typing import Dict, Literal

from automask.shot_selection import ShotSelection

# mean/std stream in one pass; median/mad need every frame (see store.py).
Reduction = Literal["mean", "std", "median", "mad"]


@dataclass(frozen=True)
class FeatureSpec:
    """One feature = a ``reduction`` applied over the frames a ``selection`` keeps.

    ``name`` is the catalogue label a stat's ``needs`` refers to (e.g. ``umean``);
    identity for caching is ``(reduction, selection)`` -- two names with the same
    reduction and selection resolve to the same cached array.
    """

    name: str
    reduction: Reduction
    selection: ShotSelection

    @property
    def content_key(self) -> str:
        """Provenance hash of what actually determines the array: the reduction
        and the full selection spec (name deliberately excluded)."""
        payload = json.dumps(
            {"reduction": self.reduction, "selection": asdict(self.selection)},
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def cache_stub(self, run: int) -> str:
        """Filename stem for this feature's cache entry (form suffix added later)."""
        return f"{self.reduction}_{self.content_key}_run{run:04d}"


FEATURES: Dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    FEATURES[spec.name] = spec
    return spec


def get_spec(name: str) -> FeatureSpec:
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {sorted(FEATURES)}")
    return FEATURES[name]
