"""The per-run arrays a masking pipeline reads.

A ``Sample`` is one run's masking context and nothing else: no ground truth, no
frozen benchmark array. Every array in it is named exactly as it is asked for --
a selected-shot reduction served by :class:`~automask.sample.image_store.ImageStore`
(``mean``, ``std``, ``mad``) or a ``psana.Detector`` calibration
accessor (``pedestals``, ``rms``, ``status_as_mask``, ...). A statistic declares
those names in its ``needs`` and reads them as attributes.

Two spaces meet here, and they are not mixed:

* reductions are ASSEMBLED -- masking happens on the assembled canvas;
* calibration constants are in psana's NATIVE PANEL form, because assembling
  scatters onto a zero-filled canvas and a constant whose zero means something
  cannot survive that. A statistic that needs one assembled interprets it first,
  then calls :func:`automask.sample.geometry.panel_to_asm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from automask.sample.image_store import REDUCTIONS, ImageStore
from automask.selection.shot_selection import ShotSelection

#: Names a statistic may declare in ``needs`` that are derived here rather than
#: loaded: see :attr:`Sample.real` and :attr:`Sample.center`.
DERIVED = ("real", "center")


@dataclass
class Sample:
    """One run's masking context: named arrays plus the selection behind them."""

    run: int
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    selection: Optional[ShotSelection] = None
    ix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    iy: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __getattr__(self, name: str) -> np.ndarray:
        arrays = self.__dict__.get("arrays") or {}
        if name in arrays:
            return arrays[name]
        raise AttributeError(
            f"run {self.__dict__.get('run')}: sample carries no array {name!r}; "
            f"loaded: {sorted(arrays)}. Add it to the statistic's `needs` so the "
            f"pipeline requests it."
        )

    @property
    def real(self) -> np.ndarray:
        """Pixels carrying a real value, from the reduction masking runs on."""
        return self.mean != 0

    def with_arrays(self, **arrays: np.ndarray) -> "Sample":
        """Copy of this sample with some arrays replaced, the rest shared.

        Used by consistency evaluation, which redraws shot-derived reductions
        while leaving calibration constants alone -- those are not estimated from
        this run's shots, so consistency folds must not perturb them.
        """
        return Sample(
            run=self.run,
            arrays={**self.arrays, **arrays},
            selection=self.selection,
            ix=self.ix,
            iy=self.iy,
        )

    def _index_maps(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ix is not None and self.iy is not None:
            return self.ix, self.iy
        from automask.sample.geometry import index_maps

        return index_maps(self.run)

    def panel_to_asm(self, panel: np.ndarray, fill=0) -> np.ndarray:
        """Scatter native data using this sample's source-bound geometry."""
        from automask.sample.geometry import assemble_panel

        return assemble_panel(panel, *self._index_maps(), fill=fill)

    def asm_to_panel(self, asm: np.ndarray) -> np.ndarray:
        """Gather assembled data using this sample's source-bound geometry."""
        from automask.sample.geometry import gather_panel

        return gather_panel(asm, *self._index_maps())

    @cached_property
    def center(self) -> Tuple[float, float]:
        """Beam center in assembled (axis0, axis1) index order."""
        from automask.sample.geometry import get_center

        return get_center(self.run)

    @classmethod
    def from_store(
        cls,
        run: int,
        selection: ShotSelection,
        needs: Iterable[str] = (),
        store: Optional[ImageStore] = None,
        gain: int = 0,
    ) -> "Sample":
        """Materialize the arrays `needs` asks for, computing or serving cached.

        A name in :data:`~automask.sample.image_store.REDUCTIONS` is reduced over the
        shots `selection` keeps; anything else is passed to psana as a detector
        calibration accessor. ``mean`` is always loaded because ``real`` -- and
        so the geometry floor -- is defined from it.
        """
        if not isinstance(selection, ShotSelection):
            raise TypeError("selection must be a ShotSelection")
        store = store or ImageStore()
        arrays = {}
        for name in {"mean", *needs} - set(DERIVED):
            arrays[name] = (
                store.reduce(run, selection, name).astype(np.float64, copy=False)
                if name in REDUCTIONS
                else store.calibration(run, name, gain=gain)
            )
        ix, iy = store.geometry(run) if hasattr(store, "geometry") else (None, None)
        return cls(
            run=int(run),
            arrays=arrays,
            selection=selection,
            ix=ix,
            iy=iy,
        )
