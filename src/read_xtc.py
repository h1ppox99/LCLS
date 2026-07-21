    try:
        i0 = psana.Detector("XPP-SB2-BMMON")           # ipm2
    except Exception:
        i0 = None

    if out is None:
        cache = os.path.join(ROOT, "src", "automask", "outputs", "cache")
        os.makedirs(cache, exist_ok=True)
        out = os.path.join(cache, f"xtc_run{run:04d}_frames.h5")

    frames, photon_eV, i0sum = [], [], []
    running_sum = None
    n = nsaved = 0