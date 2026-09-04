"""
Production psana access to calibrated xppl1016922 XTC data.

``local_run_source`` opens configured explicit files; ``slac_run_source`` uses
the standard experiment resolver. Both feed the same frame, calibration,
geometry, and profiling workflow.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from automask.io.psana1 import Psana1RunSource

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(REPO_ROOT)
XTC_DIR = str(REPO_ROOT / "xtc")
EXPERIMENT = "xppl1016922"
Backend = Literal["auto", "local", "slac"]


def _configured_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser().resolve()


def local_xtc_dir() -> Path:
    """Configured explicit-file XTC directory for off-site execution."""
    return _configured_path("AUTOMASK_XTC_DIR", REPO_ROOT / "xtc")


def local_calib_dir() -> Path:
    """Configured real-colon psana calibration directory for off-site execution."""
    return _configured_path("AUTOMASK_CALIB_DIR", REPO_ROOT / "calib")


def calib_dir(source: Psana1RunSource | None = None) -> str:
    """Calibration directory used by a source, when it has a filesystem path."""
    if source is not None and source.calib_dir is not None:
        return str(source.calib_dir)
    if source is not None and source.files:
        return str(local_calib_dir())
    psdm = os.environ.get("SIT_PSDM_DATA")
    if psdm:
        return str(Path(psdm) / "xpp" / EXPERIMENT / "calib")
    return str(local_calib_dir())


JUNGFRAU_NAME = "jungfrau1M_alcove"
JUNGFRAU_SOURCE = "XppEndstation.0:Jungfrau.0"
JUNGFRAU_CALIB_TYPE = "Jungfrau::CalibV1"


def available_xtc_runs(
    xtc_dir: str | Path | None = None,
    experiment: str = EXPERIMENT,
) -> Tuple[int, ...]:
    """Sorted runs represented by at least one local XTC stream."""
    directory = Path(xtc_dir) if xtc_dir is not None else local_xtc_dir()
    pattern = re.compile(rf"^{re.escape(experiment)}-r(?P<run>\d+)-s\d+-c\d+\.xtc$")
    runs = set()
    for path in directory.glob(f"{experiment}-r*-s*-c*.xtc"):
        match = pattern.match(path.name)
        if match:
            runs.add(int(match.group("run")))
    return tuple(sorted(runs))


def local_run_source(
    run: int = 475,
    *,
    xtc_dir: str | Path | None = None,
    calibration_dir: str | Path | None = None,
) -> Psana1RunSource:
    """Resolve one run from configured explicit XTC and calibration paths."""
    directory = Path(xtc_dir) if xtc_dir is not None else local_xtc_dir()
    files = sorted(directory.glob(f"{EXPERIMENT}-r{run:04d}-s*-c*.xtc"))
    if not files:
        raise FileNotFoundError(f"no XTC streams for run {run} in {directory}")
    calibration = calibration_dir or local_calib_dir()
    return Psana1RunSource.from_files(EXPERIMENT, run, files, calibration)


def slac_run_source(run: int, *, mpi: bool = False) -> Psana1RunSource:
    """Resolve one run through the standard SLAC psana1 experiment layout."""
    return Psana1RunSource.from_experiment(EXPERIMENT, run, smd=True, mpi=mpi)


def run_source(
    run: int,
    backend: Backend | str | None = None,
    *,
    mpi: bool = False,
) -> Psana1RunSource:
    """Resolve a run using explicit local files or the standard SLAC resolver."""
    selected = (backend or os.environ.get("AUTOMASK_BACKEND") or "auto").lower()
    if selected not in {"auto", "local", "slac"}:
        raise ValueError(
            f"AUTOMASK_BACKEND must be 'auto', 'local', or 'slac'; got {selected!r}"
        )
    if selected == "local":
        return local_run_source(run)
    if selected == "slac":
        return slac_run_source(run, mpi=mpi)
    pattern = f"{EXPERIMENT}-r{run:04d}-s*-c*.xtc"
    if any(local_xtc_dir().glob(pattern)):
        return local_run_source(run)
    return slac_run_source(run, mpi=mpi)


def _run_source(run: int, source: Psana1RunSource | None) -> Psana1RunSource:
    resolved = source or run_source(run)
    if resolved.run != run:
        raise ValueError(
            f"requested run {run}, but source describes run {resolved.run}"
        )
    return resolved


def iter_calibrated(
    run: int,
    indices,
    detname: str = JUNGFRAU_NAME,
    source: Psana1RunSource | None = None,
):
    """Pass 2: yield ``(event_index, calibrated_panel)`` for the selected events.

    ``indices`` is any iterable of event indices (as returned by
    ``ShotSelection.resolve``); frames for other events are skipped without
    decoding. Panels are ALWAYS psana-calibrated (pedestal + gain +
    common-mode) -- calibration is unconditional; any intensity normalization is
    the caller's concern. Events whose ``.calib()`` returns ``None`` are skipped.
    """
    import psana

    wanted = set(int(i) for i in indices)
    if not wanted:
        return
    last = max(wanted)
    ds = _run_source(run, source).open()
    det = psana.Detector(detname)
    for event_index, evt in enumerate(ds.events()):
        if event_index in wanted:
            panel = det.calib(evt)
            if panel is not None:
                yield event_index, np.asarray(panel, dtype=np.float32)
        if event_index >= last:
            break


def detector_calibration(
    run: int,
    constant: str,
    gain: int = 0,
    detname: str = JUNGFRAU_NAME,
    source: Psana1RunSource | None = None,
) -> np.ndarray:
    """Return one psana calibration constant for `run`, in native panel form.

    `constant` is the ``psana.Detector`` accessor name itself -- ``pedestals``,
    ``rms``, ``status_as_mask``, ``gain`` -- so no vocabulary is translated here
    and any accessor psana grows is usable immediately. Values are returned
    exactly as psana gives them; converting to this project's mask convention is
    the consuming statistic's job.
    """
    import psana

    _data_source = _run_source(run, source).open()
    detector = psana.Detector(detname)
    accessor = getattr(detector, constant, None)
    if not callable(accessor):
        raise ValueError(
            f"psana Detector for {detname!r} has no calibration accessor {constant!r}"
        )
    values = accessor(run)
    if values is None:
        raise RuntimeError(
            f"psana returned no {constant!r} calibration for {detname!r}, run {run}"
        )
    values = np.asarray(values)
    panel_shape = tuple(detector.shape(run))
    # A gain-resolved constant carries one axis more than the panels; `gain` picks the stage.
    if values.ndim == len(panel_shape) + 1:
        if not 0 <= gain < values.shape[0]:
            raise ValueError(
                f"calibration {constant!r} for run {run} has {values.shape[0]} "
                f"gain stages; got gain={gain}"
            )
        values = values[gain]
    if values.shape != panel_shape:
        raise ValueError(
            f"calibration {constant!r} for run {run} has shape {values.shape}, "
            f"expected the panel shape {panel_shape}"
        )
    return values


def panel_geometry(
    run: int,
    detname: str = JUNGFRAU_NAME,
    source: Psana1RunSource | None = None,
):
    """psana panel->assembled index maps ``(ix, iy)`` for one run (no small-data)."""
    import psana

    ds = _run_source(run, source).open()
    det = psana.Detector(detname)
    next(ds.events())  # psana needs one event before geometry
    ix = np.asarray(det.indexes_x(run), dtype=np.int64)
    iy = np.asarray(det.indexes_y(run), dtype=np.int64)
    return ix, iy
