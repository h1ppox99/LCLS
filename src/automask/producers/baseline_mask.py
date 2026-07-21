#!/usr/bin/env python3
"""
masking.py -- reproduce the xppl1016922 Jungfrau1M detector mask.

This is a clean, faithful transcription of the masking strategy in
`xpp_sharing/2_Lab6_Mask_calibration.ipynb`.  The calibration run, the event
selection, the detector image accumulation, and every hard-coded mask region
are kept identical to the notebook so that the mask produced here matches the
notebook's `Mask.npy`.  It is meant as the reference baseline we then iterate on
to improve the masking.

The strategy, in order:

  1. Load the small-data `ipm2/sum` intensity monitor for the run and keep only
     the events *below* the 99th percentile (drop the top 1% brightest shots).
  2. Loop over the raw events with psana, take the psana-assembled detector
     image `det.image(evt)` (shape (1064, 1030)), clip negative pixels to zero,
     and sum the first `n_images` selected images into `sumimg`.
  3. Build a "zero mask" from pixels that never registered signal
     (`sumimg <= 0`) and grow it by one 5x5 dilation to cover their neighbours.
  4. Add a hand-drawn geometry mask (three rectangles + a triangle) covering
     detector regions that are known-bad / must be excluded.
  5. The final mask is the union of the two.  True == masked (excluded).

Calibration / run / normalization notes (kept identical to the notebook):
  * run 475 is the Q-calibration run for this experiment.
  * NO per-shot diode normalization is applied here -- the mask step only clips
    negatives and sums (diode normalization happens later, in the processing
    step, not in mask calibration).
  * detector alias `jungfrau1M_alcove`, calibrated frames via psana.

Local-copy note
---------------
The notebook opens data with `MPIDataSource('exp=...:run=475:smd')`, which needs
the SLAC experiment registry + .smd streams.  This local copy has neither, so we
reuse `read_xtc.open_local_run()`, which hands psana the explicit stream files
and wires up the calib directory (see PSANA_XTC.md).  The per-event order is the
same, so the `ipm2` selection stays event-aligned with the psana loop.

Run it (inside the psana env -- see PSANA_XTC.md):
    source psana_env.sh
    python masking.py                 # -> Mask.npy for run 475
"""
from __future__ import annotations
import os, sys
import argparse

import numpy as np
from scipy.ndimage import binary_dilation
from matplotlib.path import Path

import h5py

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))            # .../automask/producers
AUTOMASK = os.path.dirname(HERE)                             # .../automask
sys.path.insert(0, os.path.join(os.path.dirname(AUTOMASK), "io"))  # src/io -> readers
FIGURES_DIR = os.path.join(AUTOMASK, "outputs", "figures")   # generated figures
MASKS_DIR = os.path.join(AUTOMASK, "data", "masks")          # baseline target lives here

from read_xtc import open_local_run
from lcls_xpp import smalldata_path


def _resolve_output(name: str, default_dir: str) -> str:
    """Place a bare filename in `default_dir`; keep an explicit path as-is.

    Either way the parent directory is created if needed.
    """
    parent = os.path.dirname(name) or default_dir
    os.makedirs(parent, exist_ok=True)
    return name if os.path.dirname(name) else os.path.join(default_dir, name)


def figure_path(name: str) -> str:
    """Resolve a figure filename to `automask/outputs/figures/` (bare names)."""
    return _resolve_output(name, FIGURES_DIR)


def mask_path(name: str) -> str:
    """Resolve a saved-mask filename to `automask/data/masks/` (bare names)."""
    return _resolve_output(name, MASKS_DIR)


# --- fixed parameters (identical to the notebook) --------------------------
RUN = 475                              # Q-calibration run
DETNAME = "jungfrau1M_alcove"          # psana alias for XppEndstation.0:Jungfrau.0
IMAGE_SHAPE = (1064, 1030)             # psana-assembled det.image() shape
N_IMAGES = 100                         # number of selected images to accumulate
IPM2_DROP_TOP_PERCENT = 1.0            # drop the brightest 1% of shots on ipm2
DILATION_STRUCTURE = np.ones((5, 5))   # 5x5 neighbourhood for the zero-mask grow


# ===========================================================================
#  Step 1 -- event selection from the intensity monitor
# ===========================================================================
def select_events_by_ipm2(run: int = RUN,
                          drop_top_percent: float = IPM2_DROP_TOP_PERCENT
                          ) -> np.ndarray:
    """Boolean per-event keep-array: True where the shot passes the ipm2 cut.

    Reads `ipm2/sum` (the XPP-SB2 upstream intensity monitor) from the run's
    small-data file and keeps every event whose intensity is below the
    (100 - drop_top_percent) percentile, i.e. drops the brightest
    `drop_top_percent`% of shots.
    """
    with h5py.File(smalldata_path(run), "r") as f:
        ipm2 = np.array(f["ipm2/sum"])

    thr_high = np.percentile(ipm2, 100.0 - drop_top_percent)
    keep = ipm2 < thr_high
    print(f"[ipm2] {keep.sum()}/{keep.size} events kept "
          f"(dropped top {drop_top_percent:.0f}% above {thr_high:.1f})")
    return keep


# ===========================================================================
#  Step 2 -- accumulate the summed detector image
# ===========================================================================
def accumulate_sum_image(run: int = RUN, keep_event: np.ndarray | None = None,
                        n_images: int = N_IMAGES) -> np.ndarray:
    """Sum the first `n_images` selected, negative-clipped detector images.

    For every kept event we take psana's assembled image `det.image(evt)`,
    clip negative pixels to zero, and add it to the running sum.  We stop once
    `n_images` selected images have been accumulated.
    """
    import psana

    if keep_event is None:
        keep_event = select_events_by_ipm2(run)

    ds, _ = open_local_run(run)
    det = psana.Detector(DETNAME)

    sumimg = np.zeros(IMAGE_SHAPE)
    n_used = 0
    for nevt, evt in enumerate(ds.events()):
        # Keep the selection event-aligned with the small-data ipm2 array.
        if nevt >= len(keep_event) or not keep_event[nevt]:
            continue

        img = det.image(evt)
        if img is None:
            continue

        tmp = np.copy(img)
        tmp[tmp < 0] = 0               # clip negatives before summing
        sumimg += tmp
        n_used += 1

        if n_used >= n_images:
            break

    print(f"[sum] accumulated {n_used} selected images into {sumimg.shape}")
    return sumimg


# ===========================================================================
#  Step 3-5 -- build the mask from the summed image
# ===========================================================================
def zero_mask(sumimg: np.ndarray,
             structure: np.ndarray = DILATION_STRUCTURE) -> np.ndarray:
    """Mask of pixels with no signal (`sumimg <= 0`), grown by one dilation.

    Dead / never-illuminated pixels read <= 0 in the run sum.  We dilate the
    mask so the immediate neighbours of a dead pixel are excluded too.
    """
    mask = sumimg <= 0
    return binary_dilation(mask, structure=structure)


def geometry_mask() -> np.ndarray:
    """Hand-drawn geometry mask: three rectangles + a triangle.

    These are fixed detector regions the notebook excludes by hand (panel
    edges / a bad corner next to the direct beam).  Coordinates are in
    assembled-image (row, col) pixels and are copied verbatim from the
    notebook so the result is identical.
    """
    mask = np.zeros(IMAGE_SHAPE, dtype=bool)

    # Rectangular regions (row-slice, col-slice).
    mask[970:1006, 100:1064] = True
    mask[1000:, 0:300] = True
    mask[990:1000, 0:100] = True

    # Triangular region, vertices given as (row, col).
    triangle_vertices = [(970, 100), (990, 100), (990, 50)]
    rr, cc = np.meshgrid(np.arange(IMAGE_SHAPE[0]), np.arange(IMAGE_SHAPE[1]),
                         indexing="ij")
    points = np.hstack((rr.flatten()[:, np.newaxis], cc.flatten()[:, np.newaxis]))
    triangle = Path(triangle_vertices).contains_points(points)
    triangle = triangle.reshape(IMAGE_SHAPE)

    mask |= triangle
    return mask


def build_mask(sumimg: np.ndarray) -> np.ndarray:
    """Combine the zero mask and the geometry mask. True == excluded pixel."""
    total = geometry_mask() | zero_mask(sumimg)
    print(f"[mask] {total.sum()}/{total.size} pixels masked "
          f"({100.0 * total.mean():.1f}%)")
    return total


# ===========================================================================
#  Plotting -- the same three views the notebook draws
# ===========================================================================
def plot_mask(sumimg: np.ndarray, total_mask: np.ndarray,
             out: str = "mask.png", show: bool = False):
    """Draw the notebook's mask overview: sum image | mask | overlay.

    Reproduces the three-panel figure from the calibration notebook
    (`lab6_overlapped.png`): the summed detector image, the boolean mask in
    red/white, and the mask overlaid (semi-transparent red) on the image.  Same
    display scaling as the notebook (`vmin=1.5, vmax=100`).
    """
    out = figure_path(out)             # save into src/figures/ by default

    import matplotlib
    if not show:                       # headless-safe when only saving
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Colormaps copied from the notebook: opaque red/white, and red-on-transparent.
    mask_cmap = mcolors.ListedColormap(["white", "red"])
    overlay_cmap = mcolors.ListedColormap([(1, 1, 1, 0), (1, 0, 0, 1)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    axes[0].imshow(sumimg, vmin=1.5, vmax=100)
    axes[0].set_title("summed image")
    axes[0].axis("off")

    axes[1].imshow(total_mask, cmap=mask_cmap)
    axes[1].set_title("mask (red = excluded)")
    axes[1].axis("off")

    axes[2].imshow(sumimg, vmin=1.5, vmax=100, cmap="gray")
    axes[2].imshow(total_mask, cmap=overlay_cmap, alpha=0.5)
    axes[2].set_title("overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    print(f"[plot] {out}")
    if show:
        plt.show()
    plt.close(fig)


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
