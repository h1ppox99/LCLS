#!/usr/bin/env python3
"""Build the run-level feature maps required by the recovered pipeline.

Small-data HDF5 contains cleaned first and second calibrated sums, so it can
rebuild a per-pixel mean and standard deviation without raw XTC.  The original
``umean`` was a per-shot-ipm2-normalized mean and requires a missing XTC
producer; during recovery we use the cleaned mean as an explicitly labelled
approximation.  This makes the current pipeline runnable while preserving that
provenance distinction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from automask.dataset import load_image, manifest

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "automask" / "data" / "features"
RUNS = (389, 475)


def build_run(run: int) -> None:
    """Save mean, standard deviation, and recovered umean for one run."""
    n_events = manifest()["n_events"][str(run)]
    for form in ("panel", "asm"):
        total = load_image(f"sum_calib_dropped_run{run:04d}", form).astype(np.float64)
        squared = load_image(f"sum_calib_dropped_square_run{run:04d}", form).astype(np.float64)
        mean = total / n_events
        ustd = np.sqrt(np.maximum(squared / n_events - mean * mean, 0.0))
        for name, array in (("mean", mean), ("ustd", ustd), ("umean", mean)):
            np.save(FEATURES / f"{name}_run{run:04d}_{form}.npy", array.astype(np.float32))
    print(f"[features] run {run}: mean/ustd + approximate umean")


def main() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        build_run(run)
    provenance = {
        "source": "small-data cleaned calibrated sums",
        "mean": "sum_calib_dropped / event count",
        "ustd": "sqrt(sum_calib_dropped_square / event count - mean^2)",
        "umean": "recovery approximation equal to mean; true ipm2-normalized feature needs raw-XTC producer",
    }
    (FEATURES / "README.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[done] features -> {FEATURES}")


if __name__ == "__main__":
    main()
