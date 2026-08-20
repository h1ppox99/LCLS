#!/usr/bin/env python3
"""Prepare pedestal evidence for the mask agent before Mask Step 2.

The summed diffraction image alone can hide detector defects whose intensity is
not extreme.  This step turns the gain-stage-0 pedestal into aligned, inspectable
evidence and finds *spatially coherent* pedestal residuals without using any
sample-specific coordinates.

Writes to ``--out-dir``:

  pedestal_g0_assembled.npy       gain-stage-0 pedestal in assembled coordinates
  pedestal_residual_z.npy         locally normalized, spatially smoothed residual
  pedestal_anomaly_mask.npy       candidate coherent pedestal anomalies (bool)
  pedestal_components.json        candidate component measurements and parameters
  pedestal.png                    pedestal / residual / sum-alignment preview

The candidate layer is evidence for the mask agent, not a replacement for its
layer-by-layer decision and rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_propagation, gaussian_filter, label, median_filter

ROOT = Path(__file__).resolve().parent.parent


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(finite)) or 1.0
    return center, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--background-size", type=int, default=61)
    ap.add_argument("--smooth-sigma", type=float, default=3.0)
    ap.add_argument("--seed-z", type=float, default=3.0)
    ap.add_argument("--grow-z", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--max-area", type=int, default=8000)
    ap.add_argument("--min-fill", type=float, default=0.35)
    ap.add_argument("--max-aspect", type=float, default=3.0)
    ap.add_argument("--min-asic-edge-distance", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pedestal = np.load(ROOT / "calib/ped.npy")[0].astype(np.float32)
    ix = np.load(ROOT / "calib/ix.npy").astype(np.intp)
    iy = np.load(ROOT / "calib/iy.npy").astype(np.intp)
    status = np.load(ROOT / "calib/status_bad.npy").astype(bool)

    assembled_shape = (1064, 1030)
    ped_assembled = np.full(assembled_shape, np.nan, np.float32)
    z_assembled = np.full(assembled_shape, np.nan, np.float32)
    candidate_assembled = np.zeros(assembled_shape, bool)
    ped_assembled[ix, iy] = pedestal

    components: list[dict] = []
    for panel in range(pedestal.shape[0]):
        # A wide local median removes slow ASIC gradients while retaining compact
        # circles/blobs.  Smoothing supplies power for coherent low-amplitude
        # features that are not single-pixel RMM outliers.
        background = median_filter(pedestal[panel], size=args.background_size, mode="reflect")
        residual = pedestal[panel] - background
        smooth = gaussian_filter(residual, sigma=args.smooth_sigma, mode="reflect")

        for row0 in (0, 256):
            for col0 in (0, 256, 512, 768):
                block = smooth[row0 : row0 + 256, col0 : col0 + 256]
                center, scale = _robust_center_scale(block[16:-16, 16:-16])
                z = (block - center) / scale
                raw_rows = slice(row0, row0 + 256)
                raw_cols = slice(col0, col0 + 256)
                z_assembled[ix[panel, raw_rows, raw_cols], iy[panel, raw_rows, raw_cols]] = z

                # Seed at strong residuals, then grow through adjacent moderate
                # residuals.  This is deliberately two-sided and shape-agnostic.
                seed = (np.abs(z) >= args.seed_z) & ~status[panel, raw_rows, raw_cols]
                support = (np.abs(z) >= args.grow_z) & ~status[panel, raw_rows, raw_cols]
                grown = binary_propagation(seed, mask=support)
                labs, n_labels = label(grown)
                for component_id in range(1, n_labels + 1):
                    rr, cc = np.where(labs == component_id)
                    area = int(rr.size)
                    if not (args.min_area <= area <= args.max_area):
                        continue
                    # Surface compact 2-D features here. Long lines and features
                    # attached to ASIC edges remain visible in the residual panel,
                    # but are handled by the ordinary boundary/status layers.
                    height = int(rr.max() - rr.min() + 1)
                    width = int(cc.max() - cc.min() + 1)
                    fill = area / float(height * width)
                    aspect = max(height, width) / float(min(height, width))
                    edge_distance = min(
                        int(rr.min()),
                        int(cc.min()),
                        int(255 - rr.max()),
                        int(255 - cc.max()),
                    )
                    if (
                        min(height, width) < 3
                        or fill < args.min_fill
                        or aspect > args.max_aspect
                        or edge_distance < args.min_asic_edge_distance
                    ):
                        continue

                    raw_r = row0 + rr
                    raw_c = col0 + cc
                    ass_r = ix[panel, raw_r, raw_c]
                    ass_c = iy[panel, raw_r, raw_c]
                    candidate_assembled[ass_r, ass_c] = True
                    signed_peak_index = np.argmax(np.abs(z[rr, cc]))
                    peak_z = float(z[rr[signed_peak_index], cc[signed_peak_index]])
                    components.append(
                        {
                            "id": len(components) + 1,
                            "panel": panel,
                            "asic_origin_raw": [row0, col0],
                            "area_px": area,
                            "bbox_fill_fraction": fill,
                            "bbox_aspect_ratio": aspect,
                            "asic_edge_distance_px": edge_distance,
                            "bbox_raw": [
                                int(raw_r.min()),
                                int(raw_c.min()),
                                int(raw_r.max()),
                                int(raw_c.max()),
                            ],
                            "centroid_assembled_row_col": [
                                float(ass_r.mean()),
                                float(ass_c.mean()),
                            ],
                            "bbox_assembled_row_col": [
                                int(ass_r.min()),
                                int(ass_c.min()),
                                int(ass_r.max()),
                                int(ass_c.max()),
                            ],
                            "peak_abs_z": float(np.max(np.abs(z[rr, cc]))),
                            "peak_signed_z": peak_z,
                            "polarity": "high" if peak_z > 0 else "low",
                        }
                    )

    np.save(out / "pedestal_g0_assembled.npy", ped_assembled)
    np.save(out / "pedestal_residual_z.npy", z_assembled)
    np.save(out / "pedestal_anomaly_mask.npy", candidate_assembled)

    metrics = {
        "method": "gain-stage-0 local-median residual; Gaussian spatial pooling; "
        "robust per-ASIC z; seeded connected components",
        "parameters": {
            "background_size": args.background_size,
            "smooth_sigma": args.smooth_sigma,
            "seed_abs_z": args.seed_z,
            "grow_abs_z": args.grow_z,
            "min_area_px": args.min_area,
            "max_area_px": args.max_area,
            "min_bbox_fill_fraction": args.min_fill,
            "max_bbox_aspect_ratio": args.max_aspect,
            "min_asic_edge_distance_px": args.min_asic_edge_distance,
        },
        "candidate_pixels": int(candidate_assembled.sum()),
        "component_count": len(components),
        "components": sorted(components, key=lambda item: item["peak_abs_z"], reverse=True),
    }
    (out / "pedestal_components.json").write_text(json.dumps(metrics, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sum_path = out / "sum_assembled.npy"
    summed = np.load(sum_path) if sum_path.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), constrained_layout=True)
    finite_ped = ped_assembled[np.isfinite(ped_assembled)]
    axes[0].imshow(
        ped_assembled,
        cmap="viridis",
        vmin=np.percentile(finite_ped, 1),
        vmax=np.percentile(finite_ped, 99.7),
        interpolation="nearest",
    )
    axes[0].set_title("Pedestal gain stage 0\n(aligned to summed image)")

    z_show = np.ma.masked_invalid(z_assembled)
    axes[1].imshow(z_show, cmap="coolwarm", vmin=-5, vmax=5, interpolation="nearest")
    axes[1].contour(candidate_assembled, levels=[0.5], colors="lime", linewidths=0.8)
    axes[1].set_title("Spatial pedestal residual z\n(green = candidate components)")

    if summed is not None:
        valid = summed[summed != 0]
        axes[2].imshow(
            summed, cmap="viridis", vmin=0, vmax=np.percentile(valid, 99.5), interpolation="nearest"
        )
        axes[2].contour(candidate_assembled, levels=[0.5], colors="magenta", linewidths=0.8)
        axes[2].set_title("Summed image + pedestal candidates\n(compare aligned features)")
    else:
        axes[2].imshow(candidate_assembled, cmap="gray_r", interpolation="nearest")
        axes[2].set_title("Pedestal candidate mask")
    for ax in axes:
        ax.set_xlabel("assembled col")
        ax.set_ylabel("assembled row")
    fig.savefig(out / "pedestal.png", dpi=140)
    plt.close(fig)

    print(
        f"[done] pedestal diagnostics: {len(components)} components, "
        f"{int(candidate_assembled.sum())} candidate pixels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
