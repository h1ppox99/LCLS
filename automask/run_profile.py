"""Canonical per-shot profile of one psana run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from automask.io.psana1 import Psana1RunSource
    from automask.shot_selection import ShotMeta


@dataclass
class RunProfile:
    """Discovered run contents and the per-event columns extracted from them."""

    run: int
    events: int
    payloads: List[dict]
    values: Dict[str, np.ndarray]
    summary: Dict[str, List[dict]]
    epics: List[dict]
    source: Optional["Psana1RunSource"] = field(
        default=None, repr=False, compare=False
    )
    _shot_meta: Optional["ShotMeta"] = field(
        default=None, init=False, repr=False, compare=False
    )

    def shot_meta(self) -> "ShotMeta":
        """Return the cached shot-selection view of these canonical columns."""
        if self._shot_meta is None:
            from automask.shot_selection import ShotMeta

            self._shot_meta = ShotMeta.from_profile(self)
        return self._shot_meta
