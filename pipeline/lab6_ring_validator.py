#!/usr/bin/env python3
"""Independent LaB6 ring-semantics validation backed by pyFAI.

The center fitter is allowed to propose a center and one reference radius.  This
module independently obtains the LaB6 ring-order relationships from pyFAI,
predicts every ring that intersects the assembled image, and checks whether a
ridge is observed at each predicted location.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

CALIBRANT_NAME = "LaB6_SRM660c"
WAVELENGTH_A = 1.2915
RING_HKL_LABELS = (
    "100",
    "110",
    "111",
    "200",
    "210",
    "211",
    "220",
    "300/221",
    "310",
    "311",
    "222",
    "320",
)

DEFAULT_MIN_CONTRAST = 0.03
DEFAULT_SEARCH_HALF_WIDTH_PX = 24
DEFAULT_MAX_LOCATION_ERROR_PX = 18
DEFAULT_MIN_ANGULAR_COVERAGE = 0.025
RADIAL_SIDEBAND_PX = 10.0


def load_pyfai_lab6_model(
    wavelength_A: float = WAVELENGTH_A,
    max_rings: int = len(RING_HKL_LABELS),
) -> dict:
    """Return ordered LaB6 ring relationships from pyFAI's calibrant registry."""
    try:
        import pyFAI
        from pyFAI.calibrant import CALIBRANT_FACTORY
    except ImportError as exc:
        raise RuntimeError(
            "pyFAI is required for LaB6 ring-order relationships; "
            "install the dependencies from agent/requirements.txt"
        ) from exc

    calibrant = CALIBRANT_FACTORY(CALIBRANT_NAME)
    calibrant.wavelength = wavelength_A * 1e-10
    d_spacings = np.asarray(calibrant.dspacing[:max_rings], dtype=float)
    two_theta = np.asarray(list(calibrant.get_2th())[:max_rings], dtype=float)
    if len(two_theta) < 2 or len(two_theta) != len(d_spacings):
        raise RuntimeError(f"pyFAI calibrant {CALIBRANT_NAME} returned incomplete rings")
    tan_two_theta = np.tan(two_theta)
    ratios = tan_two_theta / tan_two_theta[0]
    rings = []
    for index, (hkl, d_A, angle, ratio) in enumerate(
        zip(RING_HKL_LABELS, d_spacings, two_theta, ratios, strict=True)
    ):
        rings.append(
            {
                "index": index,
                "hkl": hkl,
                "d_spacing_A": float(d_A),
                "two_theta_rad": float(angle),
                "two_theta_deg": float(np.degrees(angle)),
                "radius_ratio_to_100": float(ratio),
            }
        )
    return {
        "provider": "pyFAI",
        "pyfai_version": str(pyFAI.version),
        "calibrant": CALIBRANT_NAME,
        "wavelength_A": float(wavelength_A),
        "rings": rings,
    }


def _smoothed_image(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    valid = np.isfinite(image) & (image != 0)
    if valid.sum() < 1000:
        raise ValueError("assembled image contains too few valid non-zero pixels")
    lo, hi = np.percentile(image[valid], [1.0, 99.8])
    clipped = np.clip(image, lo, hi)
    denominator = ndimage.gaussian_filter(valid.astype(float), sigma)
    numerator = ndimage.gaussian_filter(np.where(valid, clipped, 0.0), sigma)
    return np.where(denominator > 0.25, numerator / np.maximum(denominator, 1e-9), np.nan)


def _trimmed_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    q10, q90 = np.percentile(values, [10.0, 90.0])
    return float(np.mean(np.clip(values, q10, q90)))


def _ring_evidence(
    image: np.ndarray,
    center: tuple[float, float],
    radius_px: float,
    *,
    n_angles: int,
    min_angular_coverage: float,
) -> dict | None:
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    traces = []
    for offset in (0.0, -RADIAL_SIDEBAND_PX, RADIAL_SIDEBAND_PX):
        rr = center[0] + (radius_px + offset) * np.sin(theta)
        cc = center[1] + (radius_px + offset) * np.cos(theta)
        traces.append(
            ndimage.map_coordinates(image, [rr, cc], order=1, mode="constant", cval=np.nan)
        )
    common = np.isfinite(traces[0]) & np.isfinite(traces[1]) & np.isfinite(traces[2])
    n_valid = int(common.sum())
    if n_valid < max(6, int(np.ceil(min_angular_coverage * n_angles))):
        return None
    means = [_trimmed_mean(trace[common]) for trace in traces]
    background = 0.5 * (means[1] + means[2])
    contrast = (means[0] - background) / max(abs(background), 10.0)
    return {
        "contrast": float(contrast),
        "angular_coverage": n_valid / n_angles,
        "ring_level": means[0],
        "background_level": background,
    }


def validate_all_predicted_rings(
    image: np.ndarray,
    center: tuple[float, float],
    radius_100_px: float,
    *,
    ring_model: dict | None = None,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
    search_half_width_px: int = DEFAULT_SEARCH_HALF_WIDTH_PX,
    max_location_error_px: int = DEFAULT_MAX_LOCATION_ERROR_PX,
    min_angular_coverage: float = DEFAULT_MIN_ANGULAR_COVERAGE,
) -> dict:
    """Check every pyFAI-predicted LaB6 ring that intersects valid image pixels."""
    if ring_model is None:
        ring_model = load_pyfai_lab6_model()
    smoothed = _smoothed_image(image)
    checks = []
    for ring in ring_model["rings"]:
        predicted = radius_100_px * ring["radius_ratio_to_100"]
        at_prediction = _ring_evidence(
            smoothed,
            center,
            predicted,
            n_angles=720,
            min_angular_coverage=min_angular_coverage,
        )
        if at_prediction is None:
            checks.append(
                {
                    **ring,
                    "predicted_radius_px": float(predicted),
                    "status": "out_of_field",
                    "location_match": None,
                }
            )
            continue

        candidates = []
        for offset in range(-search_half_width_px, search_half_width_px + 1):
            evidence = _ring_evidence(
                smoothed,
                center,
                predicted + offset,
                n_angles=720,
                min_angular_coverage=min_angular_coverage,
            )
            if evidence is not None:
                candidates.append((evidence["contrast"], offset, evidence))
        _, best_offset, best_evidence = max(candidates, key=lambda item: item[0])
        has_ridge = best_evidence["contrast"] >= min_contrast
        location_match = has_ridge and abs(best_offset) <= max_location_error_px
        status = "match" if location_match else "missing" if not has_ridge else "mislocated"
        checks.append(
            {
                **ring,
                "predicted_radius_px": float(predicted),
                "observed_radius_px": float(predicted + best_offset),
                "location_error_px": float(best_offset),
                "contrast": float(best_evidence["contrast"]),
                "angular_coverage": float(best_evidence["angular_coverage"]),
                "status": status,
                "location_match": bool(location_match),
            }
        )

    in_field = [item for item in checks if item["status"] != "out_of_field"]
    failed = [item["hkl"] for item in in_field if not item["location_match"]]
    return {
        "schema_version": 1,
        "validator": "independent_all_ring_location_validator",
        "independent_evidence": (
            "Pixel evidence is re-sampled independently of the fitter's control points."
        ),
        "ring_model": ring_model,
        "candidate": {
            "center": {"row": float(center[0]), "col": float(center[1])},
            "radius_100_px": float(radius_100_px),
        },
        "thresholds": {
            "min_contrast": float(min_contrast),
            "search_half_width_px": int(search_half_width_px),
            "max_location_error_px": int(max_location_error_px),
            "min_angular_coverage": float(min_angular_coverage),
        },
        "n_in_field": len(in_field),
        "n_matched": sum(bool(item["location_match"]) for item in in_field),
        "failed_hkl": failed,
        "all_in_field_match": bool(in_field) and not failed,
        "rings": checks,
    }


def write_validation_overlay(
    image: np.ndarray,
    validation: dict,
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    valid = np.isfinite(image) & (image != 0)
    vmin, vmax = np.percentile(image[valid], [2.0, 99.5])
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    center = validation["candidate"]["center"]
    row, col = center["row"], center["col"]
    for ring in validation["rings"]:
        if ring["status"] == "out_of_field":
            continue
        color = "#3ddc84" if ring["location_match"] else "#ff453a"
        ax.add_patch(
            Circle(
                (col, row),
                ring["predicted_radius_px"],
                fill=False,
                lw=1.0,
                color=color,
                alpha=0.85,
            )
        )
        theta = np.deg2rad(320)
        label_col = col + ring["predicted_radius_px"] * np.cos(theta)
        label_row = row + ring["predicted_radius_px"] * np.sin(theta)
        if -20 <= label_col <= image.shape[1] + 20 and -20 <= label_row <= image.shape[0] + 20:
            ax.text(
                label_col,
                label_row,
                ring["hkl"],
                color=color,
                fontsize=8,
                weight="bold",
            )
    verdict = "PASS" if validation["all_in_field_match"] else "FAIL"
    ax.scatter([col], [row], marker="+", s=160, linewidths=2, color="#ffffff")
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_xlabel("assembled column (px)")
    ax.set_ylabel("assembled row (px)")
    ax.set_title(
        f"Independent pyFAI all-ring location validation: {verdict} "
        f"({validation['n_matched']}/{validation['n_in_field']})"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def _round_nested(value):
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--center", nargs=2, type=float, required=True, metavar=("ROW", "COL"))
    parser.add_argument("--radius-100-px", type=float, required=True)
    parser.add_argument("--output-prefix", default="ring_validation")
    args = parser.parse_args()

    out = Path(args.out_dir)
    image = np.load(out / "sum_assembled.npy")
    validation = _round_nested(
        validate_all_predicted_rings(
            image,
            (args.center[0], args.center[1]),
            args.radius_100_px,
        )
    )
    json_path = out / f"{args.output_prefix}.json"
    overlay_path = out / f"{args.output_prefix}_overlay.png"
    json_path.write_text(json.dumps(validation, indent=2) + "\n")
    write_validation_overlay(image, validation, overlay_path)
    verdict = "pass" if validation["all_in_field_match"] else "fail"
    print(
        f"[ring-validator] verdict={verdict} "
        f"matched={validation['n_matched']}/{validation['n_in_field']} "
        f"failed={validation['failed_hkl']} -> {json_path}"
    )
    return 0 if validation["all_in_field_match"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
