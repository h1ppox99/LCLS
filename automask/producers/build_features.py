#!/usr/bin/env python3
"""Build the run-level Jungfrau feature maps from raw XTC via ShotSelection.

Grounded replacement for the small-data version of this producer: every feature
is accumulated from psana-calibrated XTC frames chosen by an explicit
``ShotSelection`` (``automask.shot_selection``), with no small-data dependency.

Two selections per run drive the three frozen features:

* ``umean`` / ``ustd`` -- per-pixel mean and std over x-ray-**on** shots
  (the lit-beam features the window-median / black-hat statistics consume).
* ``mean``             -- per-pixel mean over x-ray-**off** shots, i.e. the
  genuine beam-off dark frame (no longer an approximation of ``umean``).

Frames are ALWAYS psana-calibrated (pedestal + gain + common-mode). The optional
``--normalization ipm2`` layers a per-shot ``median(i0)/i0`` scaling on top of
calibration for the lit build; ``none`` accumulates the calibrated frames as-is.

    source psana_env.sh
    python -m automask.producers.build_features --run 475 --n-shots 500
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from automask.io.read_xtc import iter_calibrated, panel_geometry, scan_shots
from automask.shot_selection import ShotMeta, ShotSelection

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "automask" / "data" / "features"
RUNS = (389, 475)
PANEL_SHAPE = (2, 512, 1024)
ASM_SHAPE = (1064, 1030)


def assemble(panel: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Panel -> (1064, 1030) assembled image, frozen-dataset orientation."""
    out = np.zeros(ASM_SHAPE, dtype=panel.dtype)
    out[ix, iy] = panel
    return out


def accumulate(run: int, meta: ShotMeta, selection: ShotSelection):
    """Stream the selected calibrated frames into per-pixel mean and std.

    Returns ``(mean, std, n_used, reference_i0)`` with panel-shaped arrays.
    Streams sum / sum-of-squares (one panel each), so memory is O(1) in the
    shot count -- no frame cache needed for mean/std features.
    """
    indices = selection.resolve(meta)
    normalize = selection.normalization == "ipm2"
    reference = selection.reference_intensity(meta, indices) if normalize else float("nan")

    total = np.zeros(PANEL_SHAPE, dtype=np.float64)
    squared = np.zeros(PANEL_SHAPE, dtype=np.float64)
    n_used = 0
    for event_index, panel in iter_calibrated(run, indices):
        frame = panel.astype(np.float64)
        if normalize:                      # optional per-shot i0 scaling, on top of calib
            frame *= reference / meta.intensity[event_index]
        total += frame
        squared += frame * frame
        n_used += 1
        if n_used % 50 == 0:
            print(f"[run {run:04d}] accumulated {n_used}/{indices.size} frames", flush=True)
    if n_used == 0:
        raise RuntimeError(f"run {run}: selection {selection} yielded no readable frames")
    mean = total / n_used
    std = np.sqrt(np.maximum(squared / n_used - mean * mean, 0.0))
    return mean, std, n_used, reference


def build_run(run: int, n_shots: int, filter_low: float, filter_high: float,
              normalization: str) -> dict:
    """Build umean/ustd (x-ray on) + mean (x-ray off) for one run; save both forms."""
    meta = scan_shots(run)
    ix, iy = panel_geometry(run)

    lit = ShotSelection(xray="on", n_shots=n_shots, filter_low=filter_low,
                        filter_high=filter_high, normalization=normalization)
    dark = ShotSelection(xray="off", filter_low=filter_low, filter_high=filter_high)

    umean, ustd, n_lit, ref = accumulate(run, meta, lit)
    mean, _, n_dark, _ = accumulate(run, meta, dark)

    for name, panel in (("mean", mean), ("umean", umean), ("ustd", ustd)):
        np.save(FEATURES / f"{name}_run{run:04d}_panel.npy", panel.astype(np.float32))
        np.save(FEATURES / f"{name}_run{run:04d}_asm.npy",
                assemble(panel.astype(np.float32), ix, iy))
    print(f"[features] run {run}: umean/ustd from {n_lit} x-ray-on shots, "
          f"mean from {n_dark} x-ray-off shots")
    return {"run": run, "lit_shots": n_lit, "dark_shots": n_dark,
            "lit_selection": asdict(lit), "dark_selection": asdict(dark),
            "reference_i0": None if np.isnan(ref) else ref}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, nargs="*", default=list(RUNS),
                        help="runs to build (default: 389 475)")
    parser.add_argument("--n-shots", type=int, default=500,
                        help="x-ray-on shots for the lit umean/ustd features")
    parser.add_argument("--filter-low", type=float, default=0.03,
                        help="drop this lowest fraction by intensity")
    parser.add_argument("--filter-high", type=float, default=0.03,
                        help="drop this highest fraction by intensity")
    parser.add_argument("--normalization", choices=("none", "ipm2"), default="none",
                        help="per-shot i0 scaling on top of calibration (lit build)")
    args = parser.parse_args()

    FEATURES.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in args.run:
        runs.append(build_run(run, args.n_shots, args.filter_low,
                              args.filter_high, args.normalization))
    provenance = {
        "source": "raw XTC, psana-calibrated frames selected by ShotSelection",
        "mean": "per-pixel mean over x-ray-off shots (beam-off dark frame)",
        "umean": "per-pixel mean over x-ray-on shots"
                 + ("; per-shot ipm2-normalized" if args.normalization == "ipm2" else ""),
        "ustd": "per-pixel std over the same x-ray-on shots",
        "runs": runs,
    }
    (FEATURES / "README.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[done] features -> {FEATURES}")


if __name__ == "__main__":
    main()
