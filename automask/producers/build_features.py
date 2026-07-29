#!/usr/bin/env python3
"""Prewarm the FeatureStore cache for the catalogue features (optional speedup).

Feature extraction now lives in the feature layer: a feature is a reduction over
a ShotSelection (``automask.features``), resolved lazily by the FeatureStore --
served from a warm ``.npy`` cache if present, else computed from raw XTC. The
evaluation loop materializes exactly the features a pipeline's stats declare, so
nothing here is required for correctness.

This producer just pre-computes the default catalogue (umean/ustd from beam-on
shots, mean from beam-off) for the given runs so the first evaluation is
numpy-only instead of paying the one-time XTC pass. umean and ustd share a
selection and are co-computed in a single pass.

    source psana_env.sh
    python -m automask.producers.build_features --run 389 475
"""
from __future__ import annotations

import argparse

from automask.features import FEATURES, FeatureStore, get_spec

RUNS = (389, 475)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, nargs="*", default=list(RUNS),
                        help="runs to prewarm (default: 389 475)")
    parser.add_argument("--feature", nargs="*", default=sorted(FEATURES),
                        help=f"features to prewarm (default: all -> {sorted(FEATURES)})")
    args = parser.parse_args()

    store = FeatureStore()
    for run in args.run:
        for name in args.feature:
            spec = get_spec(name)
            store.get(run, spec)   # compute+cache on miss; no-op if already warm
            over = (f"beam={spec.selection.beam!r}" if spec.source == "events"
                    else f"calib {spec.constant!r}")
            print(f"[prewarm] run {run:04d}: {name} "
                  f"({spec.reduction} over {over}) -> "
                  f"{store.path(run, spec).name}")
    print(f"[done] cache -> {store.cache_dir}")


if __name__ == "__main__":
    main()
