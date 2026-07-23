#!/usr/bin/env python3
"""Build robust IPM2-normalized Jungfrau feature maps from raw XTC.

Each valid X-ray-on calibrated frame is scaled by ``median(ipm2) / ipm2[event]``.
The per-pixel median of those frames is saved as ``umean`` and the scaled median
absolute deviation (MAD) as ``ustd``.  These are the lit-beam features used by
the window-median and black-hat masking statistics.

The producer stages normalized panel frames in an HDF5 cache, then processes
that cache in panel-row blocks.  It therefore needs disk space (about
``n * 4 MiB`` before compression) but does not hold all sampled frames in RAM.

Run 475 is the only complete local XTC run. Typical production invocation:

    source psana_env.sh
    python -m automask.producers.normalized_median --run 475 --n 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from automask.io.lcls_xpp import SmallData
from automask.io.read_xtc import JUNGFRAU_NAME, open_local_run

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "automask" / "data" / "features"
CACHE = ROOT / "automask" / "outputs" / "cache"
ASM_SHAPE = (1064, 1030)


def assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Assemble a panel map in the frozen dataset's ``(1064, 1030)`` orientation."""
    out = np.zeros(ASM_SHAPE, dtype=np.asarray(panel).dtype)
    out[np.asarray(ix, dtype=np.int64), np.asarray(iy, dtype=np.int64)] = panel
    return out


def _selection(run: int, include_xray_off: bool) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return event eligibility, IPM2 reference, and panel-to-assembled maps."""
    with SmallData(run) as sd:
        ipm2 = np.asarray(sd.i0("ipm2"), dtype=np.float64)
        xray = np.asarray(sd.xray_on(), dtype=bool)
        geo = sd.jungfrau_geometry(JUNGFRAU_NAME)
    keep = np.isfinite(ipm2) & (ipm2 > 0)
    if not include_xray_off:
        keep &= xray
    if not keep.any():
        raise RuntimeError(f"run {run} has no valid IPM2/X-ray-on events")
    return keep, float(np.median(ipm2[keep])), ipm2, geo["ix"], geo["iy"]


def stage_normalized_frames(run: int, n: int, cache_path: Path,
                            include_xray_off: bool = False) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Decode and stage the first ``n`` eligible normalized panel frames."""
    import psana

    keep, reference_i0, ipm2, ix, iy = _selection(run, include_xray_off)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds, _ = open_local_run(run)
    detector = psana.Detector(JUNGFRAU_NAME)
    staged = 0
    with h5py.File(cache_path, "w") as h5:
        frames = h5.create_dataset(
            "normalized_frames", shape=(n, 2, 512, 1024), dtype=np.float32,
            chunks=(1, 2, 64, 1024), compression="gzip", compression_opts=1,
        )
        h5.attrs.update(run=run, requested_frames=n, reference_ipm2=reference_i0,
                        selection="finite positive ipm2" + ("" if include_xray_off else "; xray_on"))
        for event_index, event in enumerate(ds.events()):
            if event_index >= len(keep):
                break
            if not keep[event_index]:
                continue
            panel = detector.calib(event)
            if panel is None:
                continue
            panel = np.asarray(panel, dtype=np.float32)
            if panel.shape != (2, 512, 1024):
                raise ValueError(f"run {run} event {event_index}: unexpected panel shape {panel.shape}")
            frames[staged] = panel * (reference_i0 / ipm2[event_index])
            staged += 1
            if staged % 50 == 0:
                print(f"[run {run:04d}] staged {staged}/{n} normalized frames", flush=True)
            if staged == n:
                break
        if staged != n:
            raise RuntimeError(f"run {run}: staged {staged}/{n} eligible XTC frames")
        h5.attrs["staged_frames"] = staged
    return ix, iy, staged, reference_i0


def robust_features(cache_path: Path, row_block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-pixel median and MAD from the staged HDF5 frames."""
    with h5py.File(cache_path, "r") as h5:
        frames = h5["normalized_frames"]
        n, modules, rows, cols = frames.shape
        median = np.empty((modules, rows, cols), dtype=np.float32)
        mad = np.empty_like(median)
        for row0 in range(0, rows, row_block):
            row1 = min(row0 + row_block, rows)
            block = frames[:, :, row0:row1, :].astype(np.float32)
            med = np.median(block, axis=0)
            median[:, row0:row1, :] = med
            mad[:, row0:row1, :] = (1.4826 * np.median(np.abs(block - med), axis=0)).astype(np.float32)
            print(f"[features] rows {row0}:{row1}/{rows}", flush=True)
    return median, mad


def save_features(run: int, median: np.ndarray, mad: np.ndarray,
                  ix: np.ndarray, iy: np.ndarray, out_dir: Path) -> None:
    """Save panel and assembled ``umean``/``ustd`` arrays for pipeline loading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, panel in (("umean", median), ("ustd", mad)):
        np.save(out_dir / f"{name}_run{run:04d}_panel.npy", panel.astype(np.float32))
        np.save(out_dir / f"{name}_run{run:04d}_asm.npy", assemble(panel, ix, iy).astype(np.float32))
    print(f"[saved] robust normalized features -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=475)
    parser.add_argument("--n", type=int, default=800, help="eligible XTC frames to use")
    parser.add_argument("--row-block", type=int, default=32, help="panel rows processed at once")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FEATURES)
    parser.add_argument("--include-xray-off", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.row_block < 1:
        parser.error("--n and --row-block must be positive")

    cache = args.cache or CACHE / f"normalized_median_run{args.run:04d}.h5"
    ix, iy, staged, reference_i0 = stage_normalized_frames(
        args.run, args.n, cache, include_xray_off=args.include_xray_off)
    median, mad = robust_features(cache, args.row_block)
    save_features(args.run, median, mad, ix, iy, args.out_dir)
    print(f"[done] run {args.run:04d}: {staged} frames; IPM2 reference {reference_i0:.3f}")
    if not args.keep_cache:
        cache.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
