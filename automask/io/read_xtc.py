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

Both of those are done for you in open_local_run() below.

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
import glob
import argparse

import numpy as np
import h5py

# --- local data locations --------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../LCLS (automask/io/ -> LCLS)
XTC_DIR = os.path.join(ROOT, "xtc")
CALIB_DIR = os.path.join(
    os.environ.get("SIT_PSDM_DATA", os.path.join(os.path.dirname(ROOT), "psdm")),
    "xpp", "xppl1016922", "calib")

JUNGFRAU_NAME = "jungfrau1M_alcove"       # psana alias; source is XppEndstation.0:Jungfrau.0


def open_local_run(run: int = 475):
    """Open all streams of a run from the local XTC, with calibration wired up.

    Returns (DataSource, list_of_files).
    """
    import psana
    # Point psana at the (real-colon) calib tree built by setup_psdm_layout.py.
    if os.path.isdir(CALIB_DIR):
        psana.setOption("psana.calib-dir", CALIB_DIR)
    else:
        print(f"[warn] calib dir not found: {CALIB_DIR}\n"
              f"       run setup_psdm_layout.py first, or frames will be uncalibrated.")
    files = sorted(glob.glob(
        os.path.join(XTC_DIR, f"xppl1016922-r{run:04d}-s0*-c00.xtc")))
    if not files:
        raise FileNotFoundError(f"no XTC streams for run {run} in {XTC_DIR}")
    print(f"[psana] opening {len(files)} streams for run {run}")
    ds = psana.DataSource(*files)
    return ds, files


def extract(run: int = 475, max_events: int = 200, out: str | None = None) -> str:
    """Read up to `max_events` calibrated frames + scalars -> HDF5. Returns path."""
    import psana
    ds, _ = open_local_run(run)
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
