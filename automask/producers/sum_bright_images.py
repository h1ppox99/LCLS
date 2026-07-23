#!/usr/bin/env python3
"""Render calibrated Jungfrau sums using only X-ray-on events.

The event selection is taken from ``lightStatus/xray`` in small-data HDF5;
frames where the X-rays were off are excluded before accumulation.  The output
is a log-scaled PNG for each requested run.  Local run 389 has only stream s00,
so its figure is explicitly labelled with the number of available frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from automask.io.lcls_xpp import SmallData
from automask.io.read_xtc import JUNGFRAU_NAME, open_local_run
from automask.producers.baseline_mask import small_data_sum


def bright_sum(run: int) -> tuple[np.ndarray, int, int, int]:
    """Return an exact X-ray-on sum by subtracting only X-ray-off frames.

    The small-data calibrated sum contains every event. Decoding every X-ray-on
    frame is valid but takes several minutes for a complete run; decode only the
    relatively few X-ray-off frames and subtract them from that full sum instead.
    """
    import psana

    with SmallData(run) as sd:
        xray_on = sd.xray_on()
    ds, _ = open_local_run(run)
    detector = psana.Detector(JUNGFRAU_NAME)
    total = small_data_sum(run).astype(np.float64)
    skipped_off = scanned = 0
    for event_index, event in enumerate(ds.events()):
        scanned = event_index + 1
        if event_index >= len(xray_on):
            break
        if xray_on[event_index]:
            continue
        image = detector.image(event)
        if image is None:
            raise RuntimeError(f"X-ray-off frame {event_index} could not be decoded for run {run}")
        total -= np.asarray(image, dtype=np.float64)
        skipped_off += 1
    if scanned != len(xray_on):
        missing = int((~xray_on[scanned:]).sum())
        raise RuntimeError(
            f"only {scanned}/{len(xray_on)} raw events are available for run {run}; "
            f"cannot remove the remaining {missing} X-ray-off frames")
    return total.astype(np.float32), int(xray_on.sum()), skipped_off, scanned


def save_plot(sum_image: np.ndarray, run: int, used: int, scanned: int, out: Path) -> None:
    """Save a robustly scaled PNG of an X-ray-on-only sum image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    positive = sum_image[sum_image > 0]
    if positive.size == 0:
        raise RuntimeError(f"run {run} sum contains no positive pixels")
    vmin = max(float(np.percentile(positive, 1)), np.finfo(np.float32).tiny)
    vmax = float(np.percentile(positive, 99.8))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(sum_image, cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
    axis.set_title(f"Run {run:04d}: X-ray-on calibrated sum ({used}/{scanned} decoded events)")
    axis.set_xlabel("assembled axis 1")
    axis.set_ylabel("assembled axis 0")
    figure.colorbar(image, ax=axis, label="summed calibrated intensity (log scale)")
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"[plot] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", default=(389, 475))
    parser.add_argument("--out-dir", type=Path, default=Path("automask/outputs/figures"))
    args = parser.parse_args()
    for run in args.runs:
        total, used, skipped_off, scanned = bright_sum(run)
        print(f"[run {run:04d}] used {used} X-ray-on frames; skipped {skipped_off} X-ray-off frames; scanned {scanned}")
        save_plot(total, run, used, scanned, args.out_dir / f"sum_xray_on_run{run:04d}.png")


if __name__ == "__main__":
    main()
