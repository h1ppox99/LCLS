#!/usr/bin/env python3
"""
read_xtc.py -- read RAW per-event data from the xppl1016922 XTC with psana.

The small-data HDF5 only stores per-event azimuthal averages + run sums.  The
XTC holds every pixel of every Jungfrau frame.  This script uses psana to pull
*calibrated* Jungfrau1M frames (pedestal + gain + common-mode corrected) event
by event and save them, plus a few per-event scalars, to a compact HDF5 you can
then load and analyse anywhere with just numpy/h5py.

It depends only on psana + numpy + h5py, so any downstream analysis can stay in
your other environment -- the same producer/consumer split smalldata_tools uses.

--------------------------------------------------------------------------
Why this opens the data the way it does (important for THIS local copy)
--------------------------------------------------------------------------
On the SLAC cluster you would normally write:

    ds = psana.DataSource('exp=xppl1016922:run=475:smd')

That resolves the run through the experiment registry and the small-data
streams.  This local copy has neither the registry entry nor the .smd streams
(only the full s00-s04 streams), so run-based discovery fails with
"no input files found".  Instead we:

  * hand psana the explicit stream files  -> psana.DataSource(*files),
  * tell psana where the calib dir is     -> psana.setOption('psana.calib-dir', ...),
    because with explicit files psana can't infer the experiment and would
    otherwise look in  $SIT_PSDM_DATA/calib  (empty) and return uncalibrated
    data (gain factors = None).

Both of those are represented by ``local_run_source()`` below.

--------------------------------------------------------------------------
Prerequisites (see PSANA_XTC.md):
    1. conda env with psana:
         conda create -p /Data/hippolyte.wallaert/envs/ana-4.0.62 \
             -c lcls-i -c conda-forge psana=4.0.62 python=3.9 numpy h5py
    2. env vars (psana needs all three):
         export SIT_PSDM_DATA=/Data/hippolyte.wallaert/psdm
         export SIT_ROOT=$SIT_PSDM_DATA/sit_root          # any valid dir
         export SIT_DATA=$SIT_PSDM_DATA/data              # holds ExpNameDb/
    3. make the local data psana-readable (once):
         python setup_psdm_layout.py
    4. run this inside the psana env:
         conda activate /Data/hippolyte.wallaert/envs/ana-4.0.62
         python read_xtc.py --events 200
--------------------------------------------------------------------------
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path

import numpy as np
import h5py

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


def extract(
    run: int = 475,
    max_events: int = 200,
    out: str | None = None,
    source: Psana1RunSource | None = None,
) -> str:
    """Read up to `max_events` calibrated frames + scalars -> HDF5. Returns path."""
    import psana
    ds = _run_source(run, source).open()
    det = psana.Detector(JUNGFRAU_NAME)
    ebeam = psana.Detector("EBeam")
    try:
        i0 = psana.Detector("XPP-SB2-BMMON")           # ipm2
    except Exception:
        i0 = None

    if out is None:
        cache = os.path.join(ROOT, "automask", "outputs", "cache")
        os.makedirs(cache, exist_ok=True)
        out = os.path.join(cache, f"xtc_run{run:04d}_frames.h5")

    frames, photon_eV, i0sum = [], [], []
    running_sum = None
    n = nsaved = 0
    for evt in ds.events():
        n += 1
        cal = det.calib(evt)                 # (2,512,1024) calibrated, or None
        if cal is None:
            continue
        cal = cal.astype(np.float32)
        running_sum = cal.copy() if running_sum is None else running_sum + cal
        if nsaved < max_events:              # keep the first N individual frames
            frames.append(cal)
            eb = ebeam.get(evt)
            photon_eV.append(eb.ebeamPhotonEnergy() if eb is not None else np.nan)
            if i0 is not None:
                d = i0.get(evt)
                i0sum.append(float(d.TotalIntensity()) if d is not None else np.nan)
            else:
                i0sum.append(np.nan)
        nsaved += 1
        if nsaved % 50 == 0:
            print(f"  {nsaved} calibrated frames (scanned {n})")
        if max_events and nsaved >= max_events:
            break

    if nsaved == 0:
        raise RuntimeError("no calibrated frames read")
    frames = np.stack(frames)
    print(f"[done] saved {len(frames)} frames, shape {frames.shape}; "
          f"summed {nsaved}")

    with h5py.File(out, "w") as h:
        h.attrs["run"] = run
        h.attrs["detname"] = JUNGFRAU_NAME
        h.create_dataset("frames", data=frames, compression="gzip", compression_opts=1)
        h.create_dataset("photon_eV", data=np.array(photon_eV))
        h.create_dataset("i0_sum", data=np.array(i0sum))
        h.create_dataset("sum_frame", data=running_sum.astype(np.float32))
        # psana's own pixel geometry (independent cross-check of small-data ix/iy)
        try:
            h.create_dataset("geom/x", data=np.asarray(det.coords_x(run)))
            h.create_dataset("geom/y", data=np.asarray(det.coords_y(run)))
            h.create_dataset("geom/ix", data=np.asarray(det.indexes_x(run)))
            h.create_dataset("geom/iy", data=np.asarray(det.indexes_y(run)))
        except Exception as e:
            print("  (psana geometry helpers unavailable:", e, ")")
    print(f"[saved] {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=475)
    ap.add_argument("--events", type=int, default=200,
                    help="number of calibrated frames to save (also used as sum length)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = extract(args.run, args.events, args.out)
    with h5py.File(path, "r") as h:
        print(f"\n{path}:")
        print(f"  frames    {h['frames'].shape}")
        print(f"  sum_frame {h['sum_frame'].shape}  mean {np.asarray(h['sum_frame']).mean():.2f}")
        print(f"  photon eV median {np.nanmedian(h['photon_eV'][()]):.1f}")
    print("\nLoad the frames with h5py and assemble them via lcls_xpp.SmallData.assemble().")
