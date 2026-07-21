        med[:, r0:r0 + 128, :] = m
        mad[:, r0:r0 + 128, :] = 1.4826 * np.median(np.abs(sl - m), axis=0)

    with h5py.File(os.path.join(SMALLDATA,
                   f"xppl1016922_Run{args.run:04d}.h5"), "r") as f:
        ix = f[f"UserDataCfg/{JUNGFRAU_NAME}/ix"][()]
        iy = f[f"UserDataCfg/{JUNGFRAU_NAME}/iy"][()]

    out = os.path.join(CACHE, f"normalized_median_run{args.run:04d}.h5")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, "w") as h:
        h.attrs["run"] = args.run