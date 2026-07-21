         max_event: int = MAX_EVENT, from_cache: bool = False, show: bool = False):
    intensity, xray_on = load_intensity(run)
    edges, bin_index = octile_bins(intensity, xray_on, N_BINS, max_event)

    plot_distribution(intensity, xray_on, edges, show=show)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"bin_intensity_run{run:04d}.npz")
    if from_cache:
        with np.load(cache) as z:
            averages, counts, edges = z["averages"], z["counts"], z["edges"]
        print(f"[cache] loaded {cache}")
    else: