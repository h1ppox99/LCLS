#!/usr/bin/env python3
"""Prewarm selected-shot images and detector calibrations for production.

    source psana_env.sh
    python -m automask.producers.build_images --run 389 475
"""
from __future__ import annotations

import argparse

from automask.image_store import REDUCTIONS, ImageStore
from automask.masking import production_pipeline
from automask.selection_presets import BEAM_ON_SELECTION

RUNS = (389, 475)


def main() -> None:
    pipeline = production_pipeline()
    needs = set(pipeline.needs())
    default_reductions = tuple(sorted(needs & REDUCTIONS))
    default_calibrations = tuple(sorted(needs - REDUCTIONS - {"real", "center"}))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", type=int, nargs="*", default=list(RUNS),
        help="runs to prewarm (default: 389 475)",
    )
    parser.add_argument(
        "--reduction", nargs="*", default=list(default_reductions),
        choices=("mean", "std", "median", "mad"),
        help=f"selected-shot reductions (default: {default_reductions})",
    )
    parser.add_argument(
        "--calibration", nargs="*", default=list(default_calibrations),
        help=f"detector constants (default: {default_calibrations})",
    )
    args = parser.parse_args()

    store = ImageStore()
    for run in args.run:
        for reduction in args.reduction:
            store.reduce(run, BEAM_ON_SELECTION, reduction)
            path = store._reduction_path(run, BEAM_ON_SELECTION, reduction, "asm")
            print(f"[prewarm] run {run:04d}: {reduction} -> {path.name}")
        for constant in args.calibration:
            store.calibration(run, constant)
            path = store._calibration_path(run, constant, 0)
            print(f"[prewarm] run {run:04d}: {constant} gain 0 -> {path.name}")
    print(f"[done] cache -> {store.cache_dir}")


if __name__ == "__main__":
    main()
