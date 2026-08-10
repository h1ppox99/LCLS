"""
features/store.py -- resolve a FeatureSpec to a per-pixel array, compute-or-cache.

This is the only psana-touching part of the feature layer. ``get(run, spec)``
returns the cached ``.npy`` if present (numpy-only, instant); otherwise it
computes the feature from raw XTC -- ``ShotMeta`` -> ``selection.resolve`` ->
``iter_calibrated`` -> reduce -> cache -> return. Evaluation therefore stays
numpy-only whenever the cache is warm, and auto-extends to any new selection or
reduction on demand.

Cache entries are keyed by ``(reduction, selection)`` content hash, not by
feature name, so ``umean`` and any other "mean over the same beam-on shots"
share one file. A mean or std request co-computes and caches BOTH (one XTC pass),
since they share a selection.

Reductions: ``mean``/``std`` stream (O(1) memory in shot count). ``median``/``mad``
are order statistics -- there is no streaming form, so the selected frames are
first staged to a temporary HDF5 cache and then reduced in panel-row blocks
(bounded RAM, ~n*4 MiB of disk). ``mad`` is the 1.4826-scaled median absolute
deviation (the robust analogue of ``std``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from automask.features.base import FeatureSpec
from automask.shot_selection import NO_NORMALIZATION, ShotMeta, ShotSelection

ROOT = Path(__file__).resolve().parents[2]
PANEL_SHAPE = (2, 512, 1024)
ASM_SHAPE = (1064, 1030)

N_GAIN = 3


def _select_line(run: int, selection: ShotSelection, counts: dict) -> str:
    """The ``[select] ...`` log line, built from a counts dict (compute or cache)."""
    n_used = counts.get("n_used")
    used = "" if n_used is None or n_used == counts.get("n_selected") else \
        f", {n_used} used"
    return (f"[select] run {run:04d}: {counts['n_events']} shots total, "
            f"{counts['n_accessible']} accessible "
            f"(beam={selection.beam}, cc={selection.cc}, vcc={selection.vcc}, "
            f"i0={selection.intensity}), "
            f"{counts['n_selected']} selected{used}")


def _assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    out = np.zeros(ASM_SHAPE, dtype=panel.dtype)
    out[ix, iy] = panel
    return out


class FeatureStore:
    """Compute-or-load feature arrays, caching provenance-keyed ``.npy`` files."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        shot_meta: ShotMeta | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else (
            ROOT / "automask" / "outputs" / "cache" / "features")
        self.shot_meta = shot_meta

    def _meta(self, run: int) -> ShotMeta:
        if self.shot_meta is None:
            from automask.io.read_xtc import scan_shots

            return scan_shots(run)
        if self.shot_meta.run != run:
            raise ValueError(
                f"FeatureStore has shot metadata for run {self.shot_meta.run}, "
                f"not run {run}"
            )
        return self.shot_meta

    # -- public ------------------------------------------------------------
    def path(self, run: int, spec: FeatureSpec, form: str = "asm") -> Path:
        return self.cache_dir / f"{spec.cache_stub(run)}_{form}.npy"

    def meta_path(self, run: int, spec: FeatureSpec) -> Path:
        return self.cache_dir / f"{spec.cache_stub(run)}_meta.json"

    def counts(self, run: int, spec: FeatureSpec) -> Optional[dict]:
        """Persisted shot counts for a cached feature, or ``None`` if unavailable
        (miss, or a cache written before sidecars existed). Keys: ``n_events``,
        ``n_accessible``, ``n_selected``, ``n_used``."""
        p = self.meta_path(run, spec)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def get(self, run: int, spec: FeatureSpec, form: str = "asm") -> np.ndarray:
        """Return the feature array for ``run``, computing + caching on a miss.

        On a hit the ``[select]`` count line is still emitted (from the sidecar),
        so cached and freshly-computed runs report the same shot bookkeeping.
        """
        target = self.path(run, spec, form)
        if not target.exists():
            self._materialize(run, spec)
        elif spec.source == "events":
            # Shot bookkeeping only means something for a reduction over events;
            # a calib constant has no shots (and no ShotSelection to describe).
            c = self.counts(run, spec)
            if c is not None:
                print(_select_line(run, spec.selection, c))
        return np.load(target)

    # -- compute -----------------------------------------------------------
    def _materialize(self, run: int, spec: FeatureSpec) -> None:
        if spec.source == "calib":
            self._materialize_calib(run, spec)
        elif spec.reduction in ("mean", "std"):
            self._materialize_mean_std(run, spec.selection)
        elif spec.reduction in ("median", "mad"):
            self._materialize_median_mad(run, spec.selection)
        else:
            raise NotImplementedError(
                f"reduction {spec.reduction!r} not implemented in FeatureStore")

    def _materialize_calib(self, run: int, spec: FeatureSpec) -> None:
        """Cache one gain stage of a psana calibration constant, both forms.

        psana resolves the calibration store and applicable run range. The
        panel->assembled maps remain the frozen project geometry used by every
        cached feature.
        """
        from automask.geometry import index_maps
        from automask.io.read_xtc import detector_calibration

        arr = detector_calibration(run, spec.constant)
        expected = (N_GAIN,) + PANEL_SHAPE
        if arr.shape != expected:
            raise ValueError(
                f"calib constant {spec.constant!r} for run {run} has shape "
                f"{arr.shape}, expected {expected}")
        panel = arr[spec.gain].astype(np.float32)
        ix, iy = index_maps(run)
        stub = spec.cache_stub(run)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / f"{stub}_panel.npy", panel)
        np.save(self.cache_dir / f"{stub}_asm.npy", _assemble(panel, ix, iy))
        (self.cache_dir / f"{stub}_meta.json").write_text(json.dumps(
            {"source": "calib", "constant": spec.constant, "gain": spec.gain,
             "run": run}))
        print(f"[calib] run {run:04d}: {spec.constant} gain {spec.gain} "
              f"-> {stub}")

    def _save_pair(self, run: int, selection: ShotSelection,
                   pairs, ix: np.ndarray, iy: np.ndarray, counts: dict) -> None:
        """Cache ``(reduction, panel)`` pairs as both panel and assembled forms,
        plus a ``_meta.json`` sidecar of shot counts (same stub as the arrays)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(counts)
        for reduction, panel in pairs:
            panel = panel.astype(np.float32)
            stub = FeatureSpec("_", reduction, selection).cache_stub(run)
            np.save(self.cache_dir / f"{stub}_panel.npy", panel)
            np.save(self.cache_dir / f"{stub}_asm.npy", _assemble(panel, ix, iy))
            (self.cache_dir / f"{stub}_meta.json").write_text(blob)

    def _materialize_mean_std(self, run: int, selection: ShotSelection) -> None:
        """One XTC pass -> cache both the mean and std for this selection."""
        from automask.io.read_xtc import panel_geometry

        mean_panel, std_panel, counts = self._accumulate(run, selection)
        ix, iy = panel_geometry(run)
        self._save_pair(run, selection,
                        (("mean", mean_panel), ("std", std_panel)), ix, iy, counts)

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
            counts = self._stage_frames(run, selection, stage_path)
            median_panel, mad_panel = self._robust_reduce(stage_path)
        finally:
            stage_path.unlink(missing_ok=True)
        ix, iy = panel_geometry(run)
        self._save_pair(run, selection,
                        (("median", median_panel), ("mad", mad_panel)), ix, iy, counts)

    def _stage_frames(self, run: int, selection: ShotSelection,
                      stage_path: Path) -> dict:
        """Decode the selected (optionally i0-normalized) frames into HDF5.

        Returns the shot-count dict (with the final ``n_used`` = frames staged)."""
        import h5py

        from automask.io.read_xtc import iter_calibrated

        meta = self._meta(run)
        indices = selection.resolve(meta)
        counts = {**selection.describe(meta), "n_selected": int(indices.size)}
        print(_select_line(run, selection, counts))
        normalize = selection.normalization != NO_NORMALIZATION
        reference = selection.reference_intensity(meta, indices) if normalize else 1.0
        i0 = meta.monitor(selection.normalization) if normalize else None

        with h5py.File(stage_path, "w") as h5:
            frames = h5.create_dataset(
                "frames", shape=(indices.size, *PANEL_SHAPE), dtype=np.float32,
                maxshape=(None, *PANEL_SHAPE), chunks=(1, 2, 64, 1024),
                compression="gzip", compression_opts=1)
            staged = 0
            for event_index, panel in iter_calibrated(run, indices):
                frame = panel.astype(np.float32)
                if normalize:
                    frame *= np.float32(reference / i0[event_index])
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
        counts["n_used"] = int(staged)
        return counts

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
                    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Stream selected calibrated frames into per-pixel (mean, std) panels.

        Returns ``(mean, std, counts)`` where ``counts['n_used']`` is the number of
        frames actually reduced (``.calib()`` misses excluded)."""
        from automask.io.read_xtc import iter_calibrated

        meta = self._meta(run)
        indices = selection.resolve(meta)
        counts = {**selection.describe(meta), "n_selected": int(indices.size)}
        print(_select_line(run, selection, counts))
        normalize = selection.normalization != NO_NORMALIZATION
        reference = selection.reference_intensity(meta, indices) if normalize else 1.0
        i0 = meta.monitor(selection.normalization) if normalize else None

        total = np.zeros(PANEL_SHAPE, dtype=np.float64)
        squared = np.zeros(PANEL_SHAPE, dtype=np.float64)
        n_used = 0
        for event_index, panel in iter_calibrated(run, indices):
            frame = panel.astype(np.float64)
            if normalize:  # optional per-shot i0 scaling, on top of calibration
                frame *= reference / i0[event_index]
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
        counts["n_used"] = int(n_used)
        return mean, std, counts
