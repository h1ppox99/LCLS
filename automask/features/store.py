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
are order statistics -- there is no streaming form, so the selected frames are
first staged to a temporary HDF5 cache and then reduced in panel-row blocks
(bounded RAM, ~n*4 MiB of disk). ``mad`` is the 1.4826-scaled median absolute
deviation (the robust analogue of ``std``), matching the ``umean``/``ustd``
convention of ``automask.producers.normalized_median``.
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
        elif spec.reduction in ("median", "mad"):
            self._materialize_median_mad(run, spec.selection)
        else:
            raise NotImplementedError(
                f"reduction {spec.reduction!r} not implemented in FeatureStore")

    def _save_pair(self, run: int, selection: ShotSelection,
                   pairs, ix: np.ndarray, iy: np.ndarray) -> None:
        """Cache ``(reduction, panel)`` pairs as both panel and assembled forms."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for reduction, panel in pairs:
            panel = panel.astype(np.float32)
            stub = FeatureSpec("_", reduction, selection).cache_stub(run)
            np.save(self.cache_dir / f"{stub}_panel.npy", panel)
            np.save(self.cache_dir / f"{stub}_asm.npy", _assemble(panel, ix, iy))

    def _materialize_mean_std(self, run: int, selection: ShotSelection) -> None:
        """One XTC pass -> cache both the mean and std for this selection."""
        from automask.io.read_xtc import panel_geometry

        mean_panel, std_panel = self._accumulate(run, selection)
        ix, iy = panel_geometry(run)
        self._save_pair(run, selection,
                        (("mean", mean_panel), ("std", std_panel)), ix, iy)

    def _materialize_median_mad(self, run: int, selection: ShotSelection) -> None:
        """Stage selected frames to disk, then cache median + scaled MAD.

        Order statistics can't stream, so this holds one XTC pass in a temporary
        HDF5 file and reduces it in panel-row blocks (bounded RAM). Median and
        MAD share the frame set and are co-computed, mirroring mean/std.
        """
        from automask.io.read_xtc import panel_geometry

        stub = FeatureSpec("_", "median", selection).cache_stub(run)
        stage_path = self.cache_dir / f"{stub}_frames.h5"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._stage_frames(run, selection, stage_path)
            median_panel, mad_panel = self._robust_reduce(stage_path)
        finally:
            stage_path.unlink(missing_ok=True)
        ix, iy = panel_geometry(run)
        self._save_pair(run, selection,
                        (("median", median_panel), ("mad", mad_panel)), ix, iy)

    def _stage_frames(self, run: int, selection: ShotSelection,
                      stage_path: Path) -> None:
        """Decode the selected (optionally i0-normalized) frames into HDF5."""
        import h5py

        from automask.io.read_xtc import iter_calibrated, scan_shots

        meta: ShotMeta = scan_shots(run)
        indices = selection.resolve(meta)
        normalize = selection.normalization == "ipm2"
        reference = selection.reference_intensity(meta, indices) if normalize else 1.0

        with h5py.File(stage_path, "w") as h5:
            frames = h5.create_dataset(
                "frames", shape=(indices.size, *PANEL_SHAPE), dtype=np.float32,
                maxshape=(None, *PANEL_SHAPE), chunks=(1, 2, 64, 1024),
                compression="gzip", compression_opts=1)
            staged = 0
            for event_index, panel in iter_calibrated(run, indices):
                frame = panel.astype(np.float32)
                if normalize:
                    frame *= np.float32(reference / meta.intensity[event_index])
                frames[staged] = frame
                staged += 1
                if staged % 50 == 0:
                    print(f"[features] run {run:04d}: staged {staged}/{indices.size} "
                          "frames", flush=True)
            if staged == 0:
                raise RuntimeError(
                    f"run {run}: selection {selection} yielded no frames")
            if staged != indices.size:          # some .calib() returned None
                frames.resize(staged, axis=0)

    @staticmethod
    def _robust_reduce(stage_path: Path, row_block: int = 32
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel median and 1.4826-scaled MAD from staged frames, block-wise."""
        import h5py

        with h5py.File(stage_path, "r") as h5:
            frames = h5["frames"]
            _, modules, rows, cols = frames.shape
            median = np.empty((modules, rows, cols), dtype=np.float32)
            mad = np.empty_like(median)
            for row0 in range(0, rows, row_block):
                row1 = min(row0 + row_block, rows)
                block = frames[:, :, row0:row1, :].astype(np.float32)
                med = np.median(block, axis=0)
                median[:, row0:row1, :] = med
                mad[:, row0:row1, :] = 1.4826 * np.median(
                    np.abs(block - med), axis=0)
                print(f"[features] rows {row0}:{row1}/{rows}", flush=True)
        return median, mad

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
