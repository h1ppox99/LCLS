#!/usr/bin/env python3
"""
synthetic/build_dataset.py -- materialize the synthetic-artifact images on disk.

Reuses the exact generation machinery of ``evaluate.py`` (source loading, the
deterministic per-example seed, the ARTIFACTS registry and parameter sampling)
but, instead of running a masker, writes each corrupted image and its injected
mask to disk. Same config + seed -> byte-identical dataset.

    python -m automask.research.synthetic.build_dataset --config automask/synthetic/config/synthetic_baseline.yaml

Layout under ``output_dir``:
    images/<tag>.npy     float32 corrupted image
    masks/<tag>.npy      bool injected mask (True == injected artifact)
    previews/<tag>.png    corrupted image + injected footprint (if --previews)
    source/<image_id>__rot<ddd>.npy   the clean rotated source, once per rotation
    manifest.csv         one row per generated example

The ground-truth hand mask is not required: injection is restricted to the
detector footprint (finite, non-zero pixels of the assembled canvas), so
artifacts never land in the dead canvas gaps. This differs from the scored
evaluation only in that footprints are not clipped around the real hand mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib

if not os.environ.get("MPLBACKEND") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automask.research.synthetic.artifacts import ARTIFACTS
from automask.research.synthetic.evaluate import (
    DEFAULT_CONFIG,
    ORIGINAL,
    HERE,
    _cases,
    _display,
    _load_source_image,
    _rotate,
    _sample_params,
)


def _valid_region(image: np.ndarray) -> np.ndarray:
    """Detector footprint used in place of ~ground_truth: assembling scatters onto
    a zero canvas, so finite non-zero pixels are the real, injectable area."""
    return np.isfinite(image) & (image != 0.0)


def _preview(corrupted, injected, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ax[0].imshow(_display(corrupted), cmap="magma")
    ax[0].set_title("corrupted")
    ax[1].imshow(injected, cmap="gray")
    ax[1].set_title("injected mask")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build(cfg: dict, previews: bool = False) -> dict:
    image, image_id = _load_source_image(cfg["image"])
    base_seed = int(cfg.get("seed", 0))
    n_per = int(cfg.get("n_per_artifact", 3))
    artifact_cfg = cfg.get("artifacts", {}) or {}
    rotations = cfg.get("rotations", [0, 90, 180, 270])
    cases = _cases()

    out_dir = os.path.abspath(
        cfg.get("output_dir", os.path.join(HERE, "..", "outputs", "synthetic_dataset"))
    )
    img_dir = os.path.join(out_dir, "images")
    msk_dir = os.path.join(out_dir, "masks")
    src_dir = os.path.join(out_dir, "source")
    fig_dir = os.path.join(out_dir, "previews")
    for d in (img_dir, msk_dir, src_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    rows = []
    src_idx = 0
    for rot_idx, degrees in enumerate(rotations):
        rimage, _ = _rotate(image, image, degrees)  # rotate image with itself
        region = _valid_region(rimage)
        src_tag = f"{image_id}__rot{int(degrees):03d}"
        np.save(os.path.join(src_dir, src_tag + ".npy"), rimage.astype(np.float32))

        for type_idx, name in enumerate(cases):
            # ORIGINAL is the clean source (already saved); skip types the config
            # does not declare, but keep type_idx aligned with evaluate.py seeds.
            if name == ORIGINAL or name not in artifact_cfg:
                continue
            acfg = dict(artifact_cfg.get(name) or {})
            allowed = acfg.pop("rotations", None)
            if allowed is not None and int(degrees) not in [int(a) for a in allowed]:
                continue
            for i in range(n_per):
                seed = (
                    base_seed
                    + src_idx * 1_000_000
                    + rot_idx * 100_000
                    + type_idx * 10_000
                    + i
                )
                rng = np.random.default_rng(seed)
                params = _sample_params(rng, acfg)
                corrupted, injected = ARTIFACTS[name](rimage, region, rng, **params)

                tag = f"{image_id}__rot{int(degrees):03d}__{name}__{i:02d}"
                img_path = os.path.join(img_dir, tag + ".npy")
                msk_path = os.path.join(msk_dir, tag + ".npy")
                np.save(img_path, corrupted.astype(np.float32))
                np.save(msk_path, injected)
                if previews:
                    _preview(
                        corrupted,
                        injected,
                        f"{tag}  seed={seed}  inj={int(injected.sum())}",
                        os.path.join(fig_dir, tag + ".png"),
                    )
                rows.append(
                    {
                        "tag": tag,
                        "image_id": image_id,
                        "rotation": int(degrees),
                        "artifact_type": name,
                        "seed": seed,
                        "injected_px": int(injected.sum()),
                        "params": json.dumps(params),
                        "image_path": os.path.relpath(img_path, out_dir),
                        "mask_path": os.path.relpath(msk_path, out_dir),
                    }
                )
                print(f"{tag:44s} seed={seed:<8d} inj={int(injected.sum()):>7d}")

    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} images -> {out_dir}")
    print(f"manifest -> {manifest}")
    return {"out_dir": out_dir, "manifest": manifest, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--previews", action="store_true", help="also write PNG previews")
    args = ap.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build(cfg, previews=args.previews)


if __name__ == "__main__":
    main()
