        if n_done % 25 == 0:
            print(f"  {n_done}/{len(want)} band frames (at event {nevt})")
        if nevt >= last_wanted:
            break

    for b, a in acc.items():
        if a["n"] == 0:
            raise RuntimeError(f"band {b} got no frames at all")
        print(f"[band {b[0]}-{b[1]}%] {a['n']} frames accumulated")
    return acc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=RUN)
    ap.add_argument("--max-event", type=int, default=MAX_EVENT,
                    help=f"ignore shots past this event (default {MAX_EVENT}; 0 = no cap)")
    args = ap.parse_args()

    bands_info = band_events(args.run, BANDS, args.max_event)
    acc = accumulate(args.run, bands_info)

    with h5py.File(os.path.join(SMALLDATA,
                   f"xppl1016922_Run{args.run:04d}.h5"), "r") as f:
        ix = f[f"UserDataCfg/{JUNGFRAU_NAME}/ix"][()]
        iy = f[f"UserDataCfg/{JUNGFRAU_NAME}/iy"][()]

    os.makedirs(FEATURES, exist_ok=True)
    meta = {"run": args.run, "max_event": args.max_event,
            "source": "extract_band_features.py", "bands": {}}

    for (lo, hi), a in acc.items():
        n = a["n"]
        mean = a["sum"] / n
        var = np.maximum(a["sumsq"] / n - mean * mean, 0.0)   # clip fp noise at 0
        ustd = np.sqrt(var)
        total = a["sum"]                                       # band sum image

        tag = f"run{args.run:04d}_p{lo:02d}_{hi:02d}"
        for name, panel in [("mean", mean), ("ustd", ustd), ("sum", total)]:
            p32 = panel.astype(np.float32)
            np.save(os.path.join(FEATURES, f"{name}_{tag}_panel.npy"), p32)
            np.save(os.path.join(FEATURES, f"{name}_{tag}_asm.npy"), assemble(p32, ix, iy))

        meta["bands"][f"{lo}_{hi}"] = {
            "percentile": [lo, hi],
            "ipm2_edges": bands_info[(lo, hi)]["edges"],
            "n_frames": int(n),
            "mean_median": float(np.median(mean)),
            "ustd_median": float(np.median(ustd)),
        }
        print(f"[band {lo}-{hi}%] mean median {np.median(mean):8.3f}  "
              f"ustd median {np.median(ustd):7.3f}  -> {tag}")
