"""Persist and reuse run profiles, cached on disk like :class:`ImageStore`.

A run profile is the per-event array table read from XTC -- the one expensive
psana pass behind selection and masking. This is its cache: one directory per
run under ``cache_dir``, so a later session (or a human script) reuses a profile
by run number instead of decoding the streams again. It mirrors ``ImageStore``,
which caches selected-shot reductions the same way.

Object columns (EPICS strings, enum labels, ``None``) are dictionary-encoded to
integer codes plus a small JSON dictionary, so loading a cached profile never
enables pickle.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from automask.io.psana1 import Psana1RunSource
from automask.profiling.run_profile import RunProfile

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2


class ProfileStore:
    """Save-or-load run profiles, one content directory per run."""

    def __init__(self, cache_dir: Path | None = None, *, backend: str | None = None):
        cache_root = Path(
            os.environ.get("AUTOMASK_CACHE_DIR") or PACKAGE_ROOT / "cache"
        )
        self.backend = backend
        if cache_dir is None:
            self.backend = backend or os.environ.get("AUTOMASK_BACKEND") or "auto"
            self.cache_dir = cache_root / "profiles" / self.backend
        else:
            self.cache_dir = Path(cache_dir)

    def path(self, run: int) -> Path:
        return self.cache_dir / f"run{int(run):04d}"

    def has(self, run: int) -> bool:
        return (self.path(run) / "profile.json").is_file()

    def load(self, run: int, *, source=None, backend: str | None = None) -> RunProfile:
        directory = self.path(run)
        if not (directory / "profile.json").is_file():
            raise FileNotFoundError(
                f"no cached profile for run {int(run):04d} under {self.cache_dir}"
            )
        profile = self._read(directory)
        stale_local = (
            profile.source is not None
            and profile.source.files
            and not all(path.is_file() for path in profile.source.files)
        )
        selected_backend = backend or self.backend
        if source is not None:
            profile.source = source
        elif selected_backend is not None or stale_local:
            from automask.io.read_xtc import run_source

            profile.source = run_source(run, selected_backend)
        return profile

    def try_load(
        self, run: int, *, source=None, backend: str | None = None
    ) -> RunProfile | None:
        return self.load(run, source=source, backend=backend) if self.has(run) else None

    def save(self, profile: RunProfile) -> Path:
        directory = self.path(profile.run)
        directory.mkdir(parents=True, exist_ok=True)
        arrays, categories, columns = self._encode_columns(profile)
        np.savez_compressed(directory / "values.npz", **arrays)
        (directory / "profile.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "run_profile",
                    "run": profile.run,
                    "events": profile.events,
                    "source": _source_to_dict(profile.source),
                    "columns": columns,
                    "categories": categories,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return directory

    # -- serialization: pickle-free, object columns -> integer codes ---------
    def _encode_columns(self, profile: RunProfile):
        arrays: dict[str, np.ndarray] = {}
        categories: dict[str, list[Any]] = {}
        columns: dict[str, str] = {}
        for index, (name, raw) in enumerate(sorted(profile.values.items())):
            values = np.asarray(raw)
            if values.ndim != 1 or values.shape[0] != profile.events:
                raise ValueError(
                    f"profile field {name!r} has shape {values.shape}, "
                    f"expected ({profile.events},)"
                )
            key = f"column_{index:04d}"
            if values.dtype.kind == "O":
                arrays[key], categories[key] = _encode_categories(values)
            else:
                arrays[key] = values
            columns[name] = key
        return arrays, categories, columns

    def _read(self, directory: Path) -> RunProfile:
        manifest = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "run_profile":
            raise ValueError(f"not a run profile artifact: {directory}")
        version = manifest.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported run profile schema_version {version!r}")
        columns = manifest.get("columns")
        categories = manifest.get("categories", {})
        if not isinstance(columns, dict) or not isinstance(categories, dict):
            raise ValueError(f"invalid run profile manifest: {directory}")
        events = int(manifest["events"])
        values = {}
        with np.load(directory / "values.npz", allow_pickle=False) as archive:
            for name, key in columns.items():
                if key not in archive:
                    raise ValueError(f"profile field {name!r} is missing array {key!r}")
                array = archive[key]
                if key in categories:
                    array = _decode_categories(
                        array, categories[key], name=name, events=events
                    )
                values[name] = array
        profile = RunProfile(
            run=int(manifest["run"]),
            events=events,
            payloads=[],
            values=values,
            summary={"xtc": [], "epics": []},
            epics=[],
            source=_source_from_dict(manifest.get("source")),
        )
        for name in profile.values:
            profile.column(name)
        return profile


def _encode_categories(values: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    """Dictionary-encode one object column without enabling pickle in NPZ."""
    dictionary: list[Any] = []
    indices: dict[str, int] = {}
    codes = np.empty(values.shape[0], dtype=np.int32)
    for index, item in enumerate(values.tolist()):
        encoded = _jsonable(item)
        signature = json.dumps(encoded, sort_keys=True, separators=(",", ":"))
        if signature not in indices:
            indices[signature] = len(dictionary)
            dictionary.append(encoded)
        codes[index] = indices[signature]
    return codes, dictionary


def _decode_categories(
    codes: np.ndarray, dictionary: list[Any], *, name: str, events: int
) -> np.ndarray:
    if codes.ndim != 1 or codes.shape[0] != events:
        raise ValueError(
            f"profile field {name!r} has category-code shape {codes.shape}, "
            f"expected ({events},)"
        )
    if codes.dtype.kind not in "iu":
        raise ValueError(f"profile field {name!r} has non-integer category codes")
    if codes.size and (codes.min() < 0 or codes.max() >= len(dictionary)):
        raise ValueError(f"profile field {name!r} has invalid category codes")
    decoded = [_decode_object(dictionary[int(code)]) for code in codes]
    return np.asarray(decoded, dtype=object)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    return value


def _decode_object(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes__"}:
        return base64.b64decode(value["__bytes__"])
    if isinstance(value, dict) and set(value) == {"__complex__"}:
        return complex(*value["__complex__"])
    return value


def _source_to_dict(source: Psana1RunSource | None) -> dict | None:
    if source is None:
        return None
    return {
        "experiment": source.experiment,
        "run": source.run,
        "files": [str(path) for path in source.files],
        "calib_dir": str(source.calib_dir) if source.calib_dir is not None else None,
        "smd": source.smd,
        "mpi": source.mpi,
    }


def _source_from_dict(value: dict | None) -> Psana1RunSource | None:
    if value is None:
        return None
    return Psana1RunSource(
        experiment=value["experiment"],
        run=int(value["run"]),
        files=tuple(Path(path) for path in value.get("files", ())),
        calib_dir=(
            Path(value["calib_dir"]) if value.get("calib_dir") is not None else None
        ),
        smd=bool(value.get("smd", False)),
        mpi=bool(value.get("mpi", False)),
    )
