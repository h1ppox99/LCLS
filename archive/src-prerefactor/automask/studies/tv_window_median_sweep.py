    SHARED_W, SHARED_K = 1.0, 5.0
    for run in RUNS:
        s = stash[run]
        no_wmed_iou = score(s["base_combo"], s["human"])["iou"]   # combo WITHOUT window_median at all
        pad_iou = base[run]["combo"]["iou"]                       # combo WITH today's pad_mask default
        M = tv_mask(s["z"], s["real"], SHARED_W, SHARED_K, pad=True)
        sc = score(s["base_combo"] | M, s["human"])
        st = score(M & ~s["geom"], s["target"])
        print(f"run {run}: weight={SHARED_W}, k={SHARED_K}  combo IoU {sc['iou']:.4f}  "
              f"T-prec {st['precision']:.3f} T-rec {st['recall']:.3f}")
        print(f"    vs no window_median  ({no_wmed_iou:.4f}): delta {sc['iou']-no_wmed_iou:+.4f}")
        print(f"    vs pad_mask default  ({pad_iou:.4f}): delta {sc['iou']-pad_iou:+.4f}")


if __name__ == "__main__":
    main()
