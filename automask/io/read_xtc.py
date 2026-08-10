"""
Production psana access to calibrated xppl1016922 XTC data.

``local_run_source`` opens explicit files in this repository; ``slac_run_source``
uses the standard experiment resolver. Both feed the same frame, calibration,
geometry, and profiling workflow.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np

from automask.io.psana1 import Psana1RunSource

# --- local data locations --------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../LCLS (automask/io/ -> LCLS)
XTC_DIR = os.path.join(ROOT, "xtc")
EXPERIMENT = "xppl1016922"


def calib_dir() -> str:
    """Resolve the experiment calib tree from ``SIT_PSDM_DATA`` *at call time*.
    """
    return os.path.join(
        os.environ.get("SIT_PSDM_DATA", os.path.join(os.path.dirname(ROOT), "psdm")),
        "xpp", "xppl1016922", "calib")

JUNGFRAU_NAME = "jungfrau1M_alcove"       # psana alias; source is XppEndstation.0:Jungfrau.0

CALIBRATION_ACCESSORS = {
    "pedestals": "pedestals",
    "pixel_rms": "rms",
}


def local_run_source(run: int = 475) -> Psana1RunSource:
    """Resolve one run from the explicit XTC files in this repository."""
    files = sorted(Path(XTC_DIR).glob(f"{EXPERIMENT}-r{run:04d}-s*-c*.xtc"))
    if not files:
        raise FileNotFoundError(f"no XTC streams for run {run} in {XTC_DIR}")
    return Psana1RunSource.from_files(EXPERIMENT, run, files, calib_dir())


def slac_run_source(run: int, *, mpi: bool = False) -> Psana1RunSource:
    """Resolve one run through the standard SLAC psana1 experiment layout."""
    return Psana1RunSource.from_experiment(EXPERIMENT, run, smd=True, mpi=mpi)


def _run_source(run: int, source: Psana1RunSource | None) -> Psana1RunSource:
    resolved = source or local_run_source(run)
    if resolved.run != run:
        raise ValueError(f"requested run {run}, but source describes run {resolved.run}")
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
    detname: str = JUNGFRAU_NAME,
    source: Psana1RunSource | None = None,
) -> np.ndarray:
    """Return a detector calibration constant through psana's official API."""
    try:
        accessor = CALIBRATION_ACCESSORS[constant]
    except KeyError as error:
        supported = ", ".join(sorted(CALIBRATION_ACCESSORS))
        raise ValueError(
            f"unsupported calibration constant {constant!r}; expected {supported}"
        ) from error

    import psana

    _data_source = _run_source(run, source).open()
    detector = psana.Detector(detname)
    values = getattr(detector, accessor)(run)
    if values is None:
        raise RuntimeError(
            f"psana returned no {constant!r} calibration for {detname!r}, run {run}"
        )
    return np.asarray(values)


def panel_geometry(
    run: int,
    detname: str = JUNGFRAU_NAME,
    source: Psana1RunSource | None = None,
):
    """psana panel->assembled index maps ``(ix, iy)`` for one run (no small-data)."""
    import psana

    ds = _run_source(run, source).open()
    det = psana.Detector(detname)
    next(ds.events())                     # psana needs one event before geometry
    ix = np.asarray(det.indexes_x(run), dtype=np.int64)
    iy = np.asarray(det.indexes_y(run), dtype=np.int64)
    return ix, iy
