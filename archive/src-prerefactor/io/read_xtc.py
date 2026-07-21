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