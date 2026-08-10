#!/usr/bin/env python3
"""Build the recovered manual Jungfrau1M masking baseline.

The baseline mirrors the lab notebook's mask recipe: mark non-positive pixels
in an accumulated calibrated image, dilate them with a 5x5 neighbourhood, and
add the three fixed geometry regions.  Masks use ``True == masked``.

The input reproduces the notebook: select events using ``ipm2/sum``, sum the
first selected calibrated XTC images, then mask them.  It requires the psana
environment and complete local XTC streams.

Run ``python -m automask.producers.baseline_mask --help`` for options.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from automask.io.lcls_xpp import SmallData

RUN = 475
DETNAME = "jungfrau1M_alcove"
ASM_SHAPE = (1064, 1030)
DILATION_STRUCTURE = np.ones((5, 5), dtype=bool)


def select_events_by_i0(run: int = RUN, drop_top_percent: float = 1.0,
                        monitor: str = "ipm2") -> np.ndarray:
    """Keep events below the top ``drop_top_percent`` of an intensity monitor.

    Defaults to ``ipm2`` because this producer reproduces the lab notebook, which
    used it. Note that ipm2 sits *upstream* of the CC/VCC beam split and so does
    not track the flux reaching the detector -- new work should prefer a
    downstream monitor such as ``sample_diode`` (see DATA.md).
    """
    with SmallData(run) as sd:
        i0 = sd.i0(monitor)
    threshold = np.percentile(i0, 100.0 - drop_top_percent)
    keep = i0 < threshold
    print(f"[{monitor}] {keep.sum()}/{keep.size} events kept "
          f"(threshold {threshold:.4g})")
    return keep


def xtc_sum(run: int = RUN, n_images: int = 100) -> np.ndarray:
    """Recreate the notebook's selected, negative-clipped XTC accumulation."""
    import psana
    from automask.io.read_xtc import local_run_source

    keep = select_events_by_i0(run)
    ds = local_run_source(run).open()
    detector = psana.Detector(DETNAME)
    total = np.zeros(ASM_SHAPE, dtype=np.float64)
    used = 0
    for event_index, event in enumerate(ds.events()):
        if event_index >= len(keep) or not keep[event_index]:
            continue
        image = detector.image(event)
        if image is None:
            continue
        image = np.maximum(np.asarray(image), 0)
        if image.shape != ASM_SHAPE:
            raise ValueError(f"psana image has shape {image.shape}; expected {ASM_SHAPE}")
        total += image
        used += 1
        if used >= n_images:
            break
    if used != n_images:
        raise RuntimeError(f"only accumulated {used}/{n_images} selected XTC images")
    print(f"[xtc] accumulated {used} selected images")
    return total.astype(np.float32)


def geometry_mask(shape: tuple[int, int] = ASM_SHAPE) -> np.ndarray:
    """Notebook's three rectangular regions plus its triangular corner."""
    if shape != ASM_SHAPE:
        raise ValueError(f"baseline geometry is defined only for {ASM_SHAPE}, got {shape}")
    mask = np.zeros(shape, dtype=bool)
    mask[970:1006, 100:1064] = True
    mask[1000:, 0:300] = True
    mask[990:1000, 0:100] = True

    # Triangle vertices are (axis-0, axis-1); avoid matplotlib for this simple
    # right triangle so the mask is buildable in the minimal numerical env.
    rows, cols = np.indices(shape)
    mask |= ((rows >= 970) & (rows < 990) & (cols >= 50) & (cols < 100)
             & (5 * rows + 2 * cols >= 5050))
    return mask


def zero_mask(sum_image: np.ndarray) -> np.ndarray:
    """Dilated mask of pixels that never registered positive signal."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(np.asarray(sum_image) <= 0, structure=DILATION_STRUCTURE)


def build_mask(sum_image: np.ndarray) -> np.ndarray:
    """Build the union of zero/dead pixels and the notebook geometry mask."""
    mask = zero_mask(sum_image) | geometry_mask(sum_image.shape)
    print(f"[mask] {mask.sum()}/{mask.size} pixels masked ({mask.mean():.2%})")
    return mask


def save_baseline(mask: np.ndarray, out: str | Path) -> Path:
    """Save ``mask`` and create its parent directory if necessary."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(bool))
    print(f"[saved] {path}")
    return path


def save_plot(mask: np.ndarray, out: str | Path, title: str) -> Path:
    """Save a headless-safe visualisation of one baseline mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(mask, cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
    axis.set_title(f"{title} — {mask.mean():.2%} masked")
    axis.set_xlabel("assembled axis 1")
    axis.set_ylabel("assembled axis 0")
    fig.colorbar(image, ax=axis, ticks=(0, 1), label="masked")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=RUN)
    parser.add_argument("--n-images", type=int, default=100, help="XTC images to accumulate")
    parser.add_argument("--out", type=Path, default=None,
                        help="mask .npy output path (default: run-specific)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="PNG output path (default: run-specific); pass '' to skip")
    args = parser.parse_args()

    image = xtc_sum(args.run, args.n_images)
    # Run 475 keeps the canonical name consumed as ground truth by
    # extract_dataset.py; other runs get a run-specific file so they don't clobber it.
    default_out = (Path("automask/data/masks/human_Mask_source.npy") if args.run == RUN
                   else Path(f"automask/data/masks/human_Mask_source_run{args.run:04d}.npy"))
    default_plot = Path(f"automask/outputs/figures/reference_mask_run{args.run:04d}.png")
    mask = build_mask(image)
    save_baseline(mask, args.out or default_out)
    plot = default_plot if args.plot is None else args.plot
    if str(plot):
        save_plot(mask, plot, f"Jungfrau1M baseline (xtc, run {args.run:04d})")


if __name__ == "__main__":
    main()
