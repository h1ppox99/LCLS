"""Compute and cache selected-shot detector images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from automask.run_profile import RunProfile
from automask.shot_selection import ShotSelection

ROOT = Path(__file__).resolve().parents[1]

Reduction = Literal["mean", "std", "mad"]
Form = Literal["asm", "panel"]
FoldStrategy = Literal["round_robin", "chronological"]

#: Reductions the store computes over selected shots. Anything a statistic needs
#: that is not one of these is a psana calibration accessor name.
REDUCTIONS = frozenset(("mean", "std", "mad"))

_FORMS = frozenset(("asm", "panel"))


def _content_key(selection: ShotSelection, reduction: Reduction) -> str:
    payload = {"reduction": reduction, "selection": asdict(selection)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _calibration_content_key(constant: str, gain: int) -> str:
    payload = {"source": "calib", "constant": constant, "gain": gain}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _reduction_stub(run: int, selection: ShotSelection, reduction: Reduction) -> str:
    return f"{reduction}_{_content_key(selection, reduction)}_run{run:04d}"


def _fold_stub(
    run: int,
    selection: ShotSelection,
    reduction: Reduction,
    strategy: FoldStrategy,
    index: int,
    n_folds: int,
) -> str:
    return (
        f"{_reduction_stub(run, selection, reduction)}_{strategy}"
        f"_fold-{index:02d}-of-{n_folds:02d}"
    )


def _calibration_stub(run: int, constant: str, gain: int) -> str:
    key = _calibration_content_key(constant, gain)
    return f"{constant}g{gain}_{key}_run{run:04d}"


def _select_line(run: int, selection: ShotSelection, counts: dict) -> str:
    n_used = counts.get("n_used")
    used = (
        ""
        if n_used is None or n_used == counts.get("n_selected")
        else f", {n_used} used"
    )
    fields = ",".join(condition.field for condition in selection.where) or "none"
    trim = selection.trim.field if selection.trim is not None else "none"
    return (
        f"[select] run {run:04d}: {counts['n_events']} shots total, "
        f"{counts['n_eligible']} eligible "
        f"(fields={fields}, trim={trim}), "
        f"{counts['n_selected']} selected{used}"
    )


def _assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Scatter panels onto the canvas the run's own index maps require."""
    out = np.zeros((int(ix.max()) + 1, int(iy.max()) + 1), dtype=panel.dtype)
    out[ix, iy] = panel
    return out


class ImageStore:
    """Compute-or-load reductions and detector calibration images."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        run_profile: RunProfile | None = None,
    ):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else (ROOT / "automask" / "cache" / "images")
        )
        self._profiles = {}
        if run_profile is not None:
            self._profiles[run_profile.run] = run_profile

    def profile(self, run: int) -> RunProfile:
        """Return one canonical profile, scanning each run at most once."""
        if run not in self._profiles:
            from automask.utils import profile_run_values

            self._profiles[run] = profile_run_values(run)
        return self._profiles[run]

    @staticmethod
    def _validate_reduction(selection, reduction, form) -> None:
        if not isinstance(selection, ShotSelection):
            raise TypeError("selection must be a ShotSelection")
        if reduction not in REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {sorted(REDUCTIONS)}, got {reduction!r}"
            )
        if form not in _FORMS:
            raise ValueError(f"form must be one of {sorted(_FORMS)}, got {form!r}")

    @staticmethod
    def _validate_calibration(constant, gain) -> None:
        if not isinstance(constant, str) or not constant:
            raise ValueError("constant must be a non-empty string")
        # How many gain stages a constant has is psana's to say (see detector_calibration).
        if not isinstance(gain, int) or isinstance(gain, bool) or gain < 0:
            raise ValueError(f"gain must be a non-negative integer, got {gain!r}")

    def _reduction_path(
        self, run: int, selection: ShotSelection, reduction: Reduction, form: Form
    ) -> Path:
        return (
            self.cache_dir / f"{_reduction_stub(run, selection, reduction)}_{form}.npy"
        )

    def _reduction_meta_path(
        self, run: int, selection: ShotSelection, reduction: Reduction
    ) -> Path:
        return (
            self.cache_dir / f"{_reduction_stub(run, selection, reduction)}_meta.json"
        )

    def _calibration_path(self, run: int, constant: str, gain: int) -> Path:
        return self.cache_dir / f"{_calibration_stub(run, constant, gain)}_panel.npy"

    def _fold_path(
        self,
        run: int,
        selection: ShotSelection,
        reduction: Reduction,
        strategy: FoldStrategy,
        index: int,
        n_folds: int,
        form: Form,
    ) -> Path:
        stub = _fold_stub(run, selection, reduction, strategy, index, n_folds)
        return self.cache_dir / f"{stub}_{form}.npy"

    def reduce(
        self,
        run: int,
        selection: ShotSelection,
        reduction: Reduction,
        form: Form = "asm",
    ) -> np.ndarray:
        """Return one selected-shot reduction, computing and caching on a miss."""
        self._validate_reduction(selection, reduction, form)
        target = self._reduction_path(run, selection, reduction, form)
        if not target.exists():
            if reduction in ("mean", "std"):
                self._materialize_mean_std(run, selection)
            else:
                self._materialize_mad(run, selection)
        else:
            counts = self.counts(run, selection, reduction)
            if counts is not None:
                print(_select_line(run, selection, counts))
        return np.load(target)

    def folds(
        self,
        run: int,
        selection: ShotSelection,
        reduction: Reduction,
        form: Form = "asm",
        *,
        n_folds: int = 10,
        strategy: FoldStrategy = "round_robin",
    ) -> tuple[np.ndarray, ...]:
        """Return reductions over round-robin or chronological shot folds."""
        self._validate_folds(n_folds, strategy)
        return self._fold_reductions(run, selection, reduction, strategy, n_folds, form)

    @staticmethod
    def _validate_folds(n_folds: int, strategy: FoldStrategy) -> None:
        if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {n_folds!r}")
        if strategy not in ("round_robin", "chronological"):
            raise ValueError(f"unknown fold strategy {strategy!r}")

    def _fold_reductions(self, run, selection, reduction, strategy, n_folds, form):
        self._validate_reduction(selection, reduction, form)
        self._validate_folds(n_folds, strategy)
        paths = [
            self._fold_path(run, selection, reduction, strategy, i, n_folds, form)
            for i in range(n_folds)
        ]
        if not all(path.exists() for path in paths):
            if reduction in ("mean", "std"):
                self._materialize_fold_mean_std(run, selection, n_folds, strategy)
            else:
                self._materialize_fold_mad(run, selection, n_folds, strategy)
        return tuple(np.load(path) for path in paths)

    def calibration(self, run: int, constant: str, gain: int = 0) -> np.ndarray:
        """Return one gain stage of a psana calibration constant, in panel form.

        Panel form only, deliberately: assembling is a scatter onto a zero-filled
        canvas, which is meaningful for intensities but not for a constant whose
        zero carries meaning -- ``status_as_mask`` would read as "bad" over every
        unmapped pixel. A statistic that wants such a constant in assembled space
        interprets it first, then calls ``geometry.panel_to_asm``.
        """
        self._validate_calibration(constant, gain)
        target = self._calibration_path(run, constant, gain)
        if not target.exists():
            self._materialize_calibration(run, constant, gain)
        return np.load(target)

    def counts(
        self, run: int, selection: ShotSelection, reduction: Reduction
    ) -> Optional[dict]:
        """Return persisted shot counts for a cached reduction, when available."""
        self._validate_reduction(selection, reduction, "asm")
        path = self._reduction_meta_path(run, selection, reduction)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _materialize_calibration(self, run: int, constant: str, gain: int) -> None:
        from automask.io.read_xtc import detector_calibration

        profile = self._profiles.get(run)
        source = profile.source if profile is not None else None
        panel = detector_calibration(run, constant, gain=gain, source=source)
        stub = _calibration_stub(run, constant, gain)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / f"{stub}_panel.npy", panel)
        (self.cache_dir / f"{stub}_meta.json").write_text(
            json.dumps(
                {
                    "source": "calib",
                    "constant": constant,
                    "gain": gain,
                    "run": run,
                }
            )
        )
        print(f"[calib] run {run:04d}: {constant} gain {gain} -> {stub}")

    def _save_pair(
        self,
        run: int,
        selection: ShotSelection,
        pairs,
        ix: np.ndarray,
        iy: np.ndarray,
        counts: dict,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(counts)
        for reduction, panel in pairs:
            panel = panel.astype(np.float32)
            stub = _reduction_stub(run, selection, reduction)
            np.save(self.cache_dir / f"{stub}_panel.npy", panel)
            np.save(self.cache_dir / f"{stub}_asm.npy", _assemble(panel, ix, iy))
            (self.cache_dir / f"{stub}_meta.json").write_text(metadata)

    def _save_folds(
        self, run, selection, pairs, ix, iy, counts, n_folds, strategy
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for reduction, full, folds in pairs:
            self._save_pair(run, selection, ((reduction, full),), ix, iy, counts)
            for index, panel in enumerate(folds):
                panel = panel.astype(np.float32)
                stub = _fold_stub(run, selection, reduction, strategy, index, n_folds)
                np.save(self.cache_dir / f"{stub}_panel.npy", panel)
                np.save(self.cache_dir / f"{stub}_asm.npy", _assemble(panel, ix, iy))
                metadata = {
                    **counts,
                    "strategy": strategy,
                    "fold_index": index,
                    "n_used": counts["fold_counts"][index],
                }
                (self.cache_dir / f"{stub}_meta.json").write_text(json.dumps(metadata))

    def _materialize_mean_std(self, run: int, selection: ShotSelection) -> None:
        from automask.io.read_xtc import panel_geometry

        mean_panel, std_panel, counts = self._accumulate(run, selection)
        ix, iy = panel_geometry(run, source=self.profile(run).source)
        self._save_pair(
            run, selection, (("mean", mean_panel), ("std", std_panel)), ix, iy, counts
        )

    def _materialize_mad(self, run: int, selection: ShotSelection) -> None:
        from automask.io.read_xtc import panel_geometry

        stub = _reduction_stub(run, selection, "mad")
        stage_path = self.cache_dir / f"{stub}_frames.h5"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            counts = self._stage_frames(run, selection, stage_path)
            mad_panel = self._robust_reduce(stage_path)
        finally:
            stage_path.unlink(missing_ok=True)
        ix, iy = panel_geometry(run, source=self.profile(run).source)
        self._save_pair(
            run,
            selection,
            (("mad", mad_panel),),
            ix,
            iy,
            counts,
        )

    def _materialize_fold_mean_std(self, run, selection, n_folds, strategy) -> None:
        from automask.io.read_xtc import panel_geometry

        full, folds, counts = self._accumulate_folds(run, selection, n_folds, strategy)
        ix, iy = panel_geometry(run, source=self.profile(run).source)
        self._save_folds(
            run,
            selection,
            (
                ("mean", full[0], folds[0]),
                ("std", full[1], folds[1]),
            ),
            ix,
            iy,
            counts,
            n_folds,
            strategy,
        )

    def _materialize_fold_mad(self, run, selection, n_folds, strategy) -> None:
        from automask.io.read_xtc import panel_geometry

        stub = _reduction_stub(run, selection, "mad")
        stage_path = self.cache_dir / f"{stub}_fold_frames.h5"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            counts = self._stage_frames(
                run, selection, stage_path, n_folds=n_folds, strategy=strategy
            )
            full, folds = self._robust_reduce_folds(stage_path, n_folds)
        finally:
            stage_path.unlink(missing_ok=True)
        ix, iy = panel_geometry(run, source=self.profile(run).source)
        self._save_folds(
            run,
            selection,
            (("mad", full, folds),),
            ix,
            iy,
            counts,
            n_folds,
            strategy,
        )

    def _stage_frames(
        self,
        run: int,
        selection: ShotSelection,
        stage_path: Path,
        n_folds: int | None = None,
        strategy: FoldStrategy | None = None,
    ) -> dict:
        import h5py

        from automask.io.read_xtc import iter_calibrated

        profile = self.profile(run)
        indices = selection.resolve(profile)
        if strategy is not None:
            self._validate_folds(n_folds, strategy)
        if strategy is not None and indices.size < 2 * n_folds:
            raise RuntimeError(
                f"run {run}: consistency evaluation needs at least "
                f"{2 * n_folds} selected shots for {n_folds} folds"
            )
        counts = selection.describe(profile)
        print(_select_line(run, selection, counts))
        normalize = selection.normalization is not None
        reference = (
            selection.normalization_reference(profile, indices) if normalize else 1.0
        )
        i0 = profile.column(selection.normalization) if normalize else None
        position = {int(event): i for i, event in enumerate(indices)}
        if strategy is not None:
            fold_counts = np.zeros(n_folds, dtype=int)

        with h5py.File(stage_path, "w") as h5:
            frames = None
            staged = 0
            for event_index, panel in iter_calibrated(
                run, indices, source=profile.source
            ):
                frame = panel.astype(np.float32)
                if frames is None:
                    # Shaped by the frames psana actually decodes, not by an assumed geometry.
                    rows = frame.shape[-2]
                    frames = h5.create_dataset(
                        "frames",
                        shape=(indices.size, *frame.shape),
                        dtype=np.float32,
                        maxshape=(None, *frame.shape),
                        chunks=(1, *frame.shape[:-2], min(64, rows), frame.shape[-1]),
                        compression="gzip",
                        compression_opts=1,
                    )
                    if strategy is not None:
                        folds = h5.create_dataset(
                            "fold", shape=(indices.size,), dtype=np.int32
                        )
                if normalize:
                    frame *= np.float32(reference / i0[event_index])
                frames[staged] = frame
                if strategy is not None:
                    pos = position[int(event_index)]
                    fold = self._fold_index(pos, indices.size, n_folds, strategy)
                    folds[staged] = fold
                    fold_counts[fold] += 1
                staged += 1
                if staged % 50 == 0:
                    print(
                        f"[images] run {run:04d}: staged {staged}/{indices.size} frames",
                        flush=True,
                    )
            if staged == 0:
                raise RuntimeError(
                    f"run {run}: selection {selection} yielded no frames"
                )
            if staged != indices.size:
                frames.resize(staged, axis=0)
                if strategy is not None:
                    folds.resize(staged, axis=0)
        counts["n_used"] = int(staged)
        if strategy is not None:
            counts["fold_counts"] = fold_counts.tolist()
        return counts

    @staticmethod
    def _fold_index(position, n_selected, n_folds, strategy):
        if strategy == "round_robin":
            return position % n_folds
        return min(n_folds * position // n_selected, n_folds - 1)

    @staticmethod
    def _robust_reduce(
        stage_path: Path, row_block: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        import h5py

        with h5py.File(stage_path, "r") as h5:
            frames = h5["frames"]
            _, modules, rows, cols = frames.shape
            mad = np.empty((modules, rows, cols), dtype=np.float32)
            for row0 in range(0, rows, row_block):
                row1 = min(row0 + row_block, rows)
                block = frames[:, :, row0:row1, :].astype(np.float32)
                med = np.median(block, axis=0)
                mad[:, row0:row1, :] = 1.4826 * np.median(np.abs(block - med), axis=0)
                print(f"[images] rows {row0}:{row1}/{rows}", flush=True)
        return mad

    @staticmethod
    def _robust_reduce_folds(stage_path: Path, n_folds: int, row_block: int = 32):
        import h5py

        with h5py.File(stage_path, "r") as h5:
            frames = h5["frames"]
            rows_by_group = [
                np.flatnonzero(np.asarray(h5["fold"]) == i) for i in range(n_folds)
            ]
            if any(rows.size < 2 for rows in rows_by_group):
                raise RuntimeError(
                    "consistency evaluation needs at least two shots per fold"
                )
            shape = (n_folds, *frames.shape[1:])
            mad = np.empty(shape, dtype=np.float32)
            full_mad = np.empty(frames.shape[1:], dtype=np.float32)
            for row0 in range(0, frames.shape[-2], row_block):
                row1 = min(row0 + row_block, frames.shape[-2])
                block = frames[:, :, row0:row1, :].astype(np.float32)
                med = np.median(block, axis=0)
                full_mad[:, row0:row1] = 1.4826 * np.median(np.abs(block - med), axis=0)
                for group, rows in enumerate(rows_by_group):
                    block = frames[rows, :, row0:row1, :].astype(np.float32)
                    med = np.median(block, axis=0)
                    mad[group, :, row0:row1] = 1.4826 * np.median(
                        np.abs(block - med), axis=0
                    )
        return full_mad, mad

    def _accumulate(
        self, run: int, selection: ShotSelection
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        from automask.io.read_xtc import iter_calibrated

        profile = self.profile(run)
        indices = selection.resolve(profile)
        counts = selection.describe(profile)
        print(_select_line(run, selection, counts))
        normalize = selection.normalization is not None
        reference = (
            selection.normalization_reference(profile, indices) if normalize else 1.0
        )
        i0 = profile.column(selection.normalization) if normalize else None

        total = squared = None
        n_used = 0
        for event_index, panel in iter_calibrated(run, indices, source=profile.source):
            frame = panel.astype(np.float64)
            if total is None:
                # Shaped by the frames psana actually decodes, not by an assumed geometry.
                total, squared = np.zeros_like(frame), np.zeros_like(frame)
            if normalize:
                frame *= reference / i0[event_index]
            total += frame
            squared += frame * frame
            n_used += 1
            if n_used % 50 == 0:
                print(
                    f"[images] run {run:04d}: {n_used}/{indices.size} frames",
                    flush=True,
                )
        if n_used == 0:
            raise RuntimeError(f"run {run}: selection {selection} yielded no frames")
        mean = total / n_used
        std = np.sqrt(np.maximum(squared / n_used - mean * mean, 0.0))
        counts["n_used"] = int(n_used)
        return mean, std, counts

    def _accumulate_folds(self, run, selection, n_folds, strategy):
        from automask.io.read_xtc import iter_calibrated

        profile = self.profile(run)
        indices = selection.resolve(profile)
        self._validate_folds(n_folds, strategy)
        if indices.size < 2 * n_folds:
            raise RuntimeError(
                f"run {run}: consistency evaluation needs at least "
                f"{2 * n_folds} selected shots for {n_folds} folds"
            )
        counts = selection.describe(profile)
        print(_select_line(run, selection, counts))
        normalize = selection.normalization is not None
        reference = (
            selection.normalization_reference(profile, indices) if normalize else 1.0
        )
        i0 = profile.column(selection.normalization) if normalize else None
        position = {int(event): i for i, event in enumerate(indices)}

        n = np.zeros(n_folds, dtype=np.int64)
        total = squared = None
        for event_index, panel in iter_calibrated(run, indices, source=profile.source):
            frame = panel.astype(np.float64)
            if normalize:
                frame *= reference / i0[event_index]
            if total is None:
                shape = (n_folds, *frame.shape)
                total, squared = np.zeros(shape), np.zeros(shape)
            pos = position[int(event_index)]
            fold = self._fold_index(pos, indices.size, n_folds, strategy)
            n[fold] += 1
            total[fold] += frame
            squared[fold] += frame * frame
        if total is None or (n < 2).any():
            raise RuntimeError(f"run {run}: consistency fold counts {n.tolist()}")
        mean = total / n.reshape((-1,) + (1,) * (total.ndim - 1))
        var = squared / n.reshape((-1,) + (1,) * (total.ndim - 1)) - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        full_n = int(n.sum())
        full_mean = total.sum(axis=0) / full_n
        full_var = squared.sum(axis=0) / full_n - full_mean * full_mean
        counts.update(n_used=full_n, fold_counts=n.tolist())
        return ((full_mean, np.sqrt(np.maximum(full_var, 0.0))), (mean, std), counts)
