"""Canonical per-shot profile of one psana run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from automask.io.psana1 import Psana1RunSource


@dataclass
class RunProfile:
    """Discovered run contents and the per-event columns extracted from them."""

    run: int
    events: int
    payloads: List[dict]
    values: Dict[str, np.ndarray]
    summary: Dict[str, List[dict]]
    epics: List[dict]
    source: Optional["Psana1RunSource"] = field(default=None, repr=False, compare=False)

    def column(self, name: str) -> np.ndarray:
        """Return one canonical per-shot field with a concise missing-field error."""
        if name not in self.values:
            available = sorted(self.values)
            shown = available[:12]
            suffix = " ..." if len(available) > len(shown) else ""
            raise KeyError(
                f"run {self.run}: field {name!r} is unavailable; "
                f"available examples: {shown}{suffix}"
            )
        values = np.asarray(self.values[name])
        if values.ndim != 1 or values.shape[0] != self.events:
            raise ValueError(
                f"run {self.run}: field {name!r} has shape {values.shape}, "
                f"expected ({self.events},)"
            )
        return values
