"""
features/store.py -- resolve a FeatureSpec to a per-pixel array, compute-or-cache.

This is the only psana-touching part of the feature layer. ``get(run, spec)``
returns the cached ``.npy`` if present (numpy-only, instant); otherwise it
computes the feature from raw XTC -- ``scan_shots`` -> ``selection.resolve`` ->
``iter_calibrated`` -> reduce -> cache -> return. Evaluation therefore stays
numpy-only whenever the cache is warm, and auto-extends to any new selection or
reduction on demand.

Cache entries are keyed by ``(reduction, selection)`` content hash, not by
feature name, so ``umean`` and any other "mean over the same x-ray-on shots"
share one file. A mean or std request co-computes and caches BOTH (one XTC pass),
since they share a selection.

Reductions: ``mean``/``std`` stream (O(1) memory in shot count). ``median``/``mad``
need every frame held at once and are not implemented here yet -- use
``automask.producers.normalized_median`` for robust lit features meanwhile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from automask.features.base import FeatureSpec
from automask.shot_selection import ShotMeta, ShotSelection

ROOT = Path(__file__).resolve().parents[2]
PANEL_SHAPE = (2, 512, 1024)
ASM_SHAPE = (1064, 1030)


def _assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    out = np.zeros(ASM_SHAPE, dtype=panel.dtype)
    out[ix, iy] = panel
    return out


class FeatureStore:
    """Compute-or-load feature arrays, caching provenance-keyed ``.npy`` files."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else (
            ROOT / "automask" / "outputs" / "cache" / "features")

    # -- public ------------------------------------------------------------
    def path(self, run: int, spec: FeatureSpec, form: str = "asm") -> Path:
        return self.cache_dir / f"{spec.cache_stub(run)}_{form}.npy"

    def get(self, run: int, spec: FeatureSpec, form: str = "asm") -> np.ndarray:
        """Return the feature array for ``run``, computing + caching on a miss."""
        target = self.path(run, spec, form)
        if not target.exists():
            self._materialize(run, spec)
        return np.load(target)

    # -- compute -----------------------------------------------------------
    def _materialize(self, run: int, spec: FeatureSpec) -> None:
        if spec.reduction in ("mean", "std"):
            self._materialize_mean_std(run, spec.selection)
        else:
            raise NotImplementedError(
                f"reduction {spec.reduction!r} not implemented in FeatureStore; "
                f"use automask.producers.normalized_median for median/mad features")

    def _materialize_mean_std(self, run: int, selection: ShotSelection) -> None:
        """One XTC pass -> cache both the mean and std for this selection."""
        from automask.io.read_xtc import panel_geometry

        mean_panel, std_panel = self._accumulate(run, selection)
        ix, iy = panel_geometry(run)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for reduction, panel in (("mean", mean_panel), ("std", std_panel)):
            stub = FeatureSpec("_", reduction, selection).cache_stub(run)
            np.save(self.cache_dir / f"{stub}_panel.npy", panel.astype(np.float32))
            np.save(self.cache_dir / f"{stub}_asm.npy",
                    _assemble(panel.astype(np.float32), ix, iy))

    def _accumulate(self, run: int, selection: ShotSelection
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stream selected calibrated frames into per-pixel (mean, std) panels."""
        from automask.io.read_xtc import iter_calibrated, scan_shots

        meta: ShotMeta = scan_shots(run)
        indices = selection.resolve(meta)
        normalize = selection.normalization == "ipm2"
        reference = selection.reference_intensity(meta, indices) if normalize else 1.0

        total = np.zeros(PANEL_SHAPE, dtype=np.float64)
        squared = np.zeros(PANEL_SHAPE, dtype=np.float64)
        n_used = 0
        for event_index, panel in iter_calibrated(run, indices):
            frame = panel.astype(np.float64)
            if normalize:  # optional per-shot i0 scaling, on top of calibration
                frame *= reference / meta.intensity[event_index]
            total += frame
            squared += frame * frame
            n_used += 1
            if n_used % 50 == 0:
                print(f"[features] run {run:04d}: {n_used}/{indices.size} frames",
                      flush=True)
        if n_used == 0:
            raise RuntimeError(f"run {run}: selection {selection} yielded no frames")
        mean = total / n_used
        std = np.sqrt(np.maximum(squared / n_used - mean * mean, 0.0))
        return mean, std
