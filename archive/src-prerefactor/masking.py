
# ===========================================================================
#  Orchestration
# ===========================================================================
def make_mask(run: int = RUN, n_images: int = N_IMAGES,
             out: str = "human_Mask_source.npy", plot_out: str | None = None) -> np.ndarray:
    """Reproduce the full notebook mask for `run` and save it to `out`.

    If `plot_out` is given, also save the three-panel mask overview figure.
    """
    keep = select_events_by_ipm2(run)
    sumimg = accumulate_sum_image(run, keep, n_images)
    total_mask = build_mask(sumimg)
    out = mask_path(out)                # save into automask/data/masks/ by default
    np.save(out, total_mask)
    print(f"[saved] {out}")
    if plot_out is not None:
        plot_mask(sumimg, total_mask, out=plot_out)
    return total_mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=RUN)
    ap.add_argument("--n-images", type=int, default=N_IMAGES,
                    help="number of selected images to accumulate into the sum")
    ap.add_argument("--out", default="human_Mask_source.npy",
                    help="mask filename; a bare name goes into automask/data/masks/ "
                         "(default human_Mask_source.npy)")
    ap.add_argument("--plot", nargs="?", const="", default=None, metavar="PNG",
                    help="also save the sum/mask/overlay figure; a bare name goes into "
                         "automask/outputs/figures/ (default mask_run<RUN>.png)")
    args = ap.parse_args()

    # --plot with no value -> default per-run name inside src/figures/.
    plot_out = args.plot
    if plot_out == "":
        plot_out = f"mask_run{args.run:04d}.png"
    make_mask(args.run, args.n_images, args.out, plot_out=plot_out)