#!/usr/bin/env python3
"""Step 3 (deterministic executor): apply the mask agent's mask to the assembled sum.

Expects in outputs/<run>/:
  sum_assembled.npy      (1064, 1030) float64   — from step 2
  mask_assembled.npy     (1064, 1030) bool      — written by the mask agent, True = masked

Writes:
  masked_sum.npy   assembled sum with masked pixels set to NaN  (the pipeline endpoint)
  masked_sum.png   side-by-side before/after preview
  mask_log.json    coverage stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    img = np.load(out / "sum_assembled.npy")
    mask = np.load(out / "mask_assembled.npy")
    assert mask.shape == img.shape and mask.dtype == bool, "mask must be (1064,1030) bool"

    masked = img.copy()
    masked[mask] = np.nan
    np.save(out / "masked_sum.npy", masked)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = img[(img != 0) & ~mask]
    vmax = np.percentile(nz, 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    axes[0].imshow(img, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title("sum (before mask)")
    show = np.ma.masked_invalid(masked)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("red")
    axes[1].imshow(show, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    axes[1].set_title("masked sum — PIPELINE ENDPOINT (red = masked)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out / "masked_sum.png", dpi=110, bbox_inches="tight")

    in_panel = img != 0  # gap rows are exactly 0 in the assembled image
    log = {
        "mask_pixels": int(mask.sum()),
        "mask_fraction_of_image": float(mask.mean()),
        "mask_fraction_of_panels": float(mask[in_panel].mean()),
        "unmasked_pixels": int((~mask & in_panel).sum()),
    }
    (out / "mask_log.json").write_text(json.dumps(log, indent=2))
    print(
        f"[done] masked {log['mask_pixels']} px "
        f"({log['mask_fraction_of_panels'] * 100:.1f}% of panel area)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
