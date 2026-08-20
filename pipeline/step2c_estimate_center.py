#!/usr/bin/env python3
"""Physics-grounded image-center calibration for LaB6 Run0475.

The historical 367/531/734 px diffuse anchors are excluded from every center
constraint. The executor proposes two free-radius ridges, tests whether they
behave like LaB6 (100)/(110), then accepts, revises, or escalates the fit.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

try:
    from pipeline.lab6_ring_validator import (
        load_pyfai_lab6_model,
        validate_all_predicted_rings,
        write_validation_overlay,
    )
except ModuleNotFoundError:  # direct execution: python pipeline/step2c_estimate_center.py
    from lab6_ring_validator import (
        load_pyfai_lab6_model,
        validate_all_predicted_rings,
        write_validation_overlay,
    )

LATTICE_A_A = 4.15695
LAMBDA_A = 1.2915
PIXEL_MM = 0.075
HKL_N = {"100": 1, "110": 2}
UNINDEXED_ANCHORS_PX = [367, 531, 734]
RADIAL_SIDEBAND_PX = 10.0
DEFAULT_DISTANCE_BOUNDS_MM = (175.0, 220.0)


def q_hkl(n: int) -> float:
    return 2 * np.pi / LATTICE_A_A * np.sqrt(n)


def tan_two_theta(q: float) -> float:
    return float(np.tan(2 * np.arcsin(q * LAMBDA_A / (4 * np.pi))))


Q_HKL = {hkl: q_hkl(n) for hkl, n in HKL_N.items()}
TAN_RATIO_110_100 = tan_two_theta(Q_HKL["110"]) / tan_two_theta(Q_HKL["100"])


def radius_from_distance(distance_mm: float, hkl: str = "100") -> float:
    return distance_mm * tan_two_theta(Q_HKL[hkl]) / PIXEL_MM


def distance_from_radius(radius_px: float, hkl: str) -> float:
    return radius_px * PIXEL_MM / tan_two_theta(Q_HKL[hkl])


def _smoothed_image(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    valid = np.isfinite(img) & (img != 0)
    if valid.sum() < 1000:
        raise ValueError("assembled image contains too few valid non-zero pixels")
    lo, hi = np.percentile(img[valid], [1.0, 99.8])
    clipped = np.clip(img, lo, hi)
    denominator = ndimage.gaussian_filter(valid.astype(float), sigma)
    numerator = ndimage.gaussian_filter(np.where(valid, clipped, 0.0), sigma)
    return np.where(denominator > 0.25, numerator / np.maximum(denominator, 1e-9), np.nan)


def _trimmed_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    q10, q90 = np.percentile(values, [10.0, 90.0])
    return float(np.mean(np.clip(values, q10, q90)))


def _ring_relative_contrast(
    smoothed: np.ndarray,
    center: tuple[float, float],
    radius: float,
    n_angles: int,
) -> dict | None:
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    traces = []
    for offset in (0.0, -RADIAL_SIDEBAND_PX, RADIAL_SIDEBAND_PX):
        rr = center[0] + (radius + offset) * sin_t
        cc = center[1] + (radius + offset) * cos_t
        traces.append(
            ndimage.map_coordinates(smoothed, [rr, cc], order=1, mode="constant", cval=np.nan)
        )
    common = np.isfinite(traces[0]) & np.isfinite(traces[1]) & np.isfinite(traces[2])
    n_valid = int(common.sum())
    if n_valid < max(6, int(0.025 * n_angles)):
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


def _pair_score(
    smoothed: np.ndarray,
    row: float,
    col: float,
    radius_100: float,
    n_angles: int,
    ratio_110_100: float = TAN_RATIO_110_100,
) -> tuple[float, list[dict]]:
    details = []
    pairs = (("100", radius_100), ("110", radius_100 * ratio_110_100))
    for hkl, radius in pairs:
        evidence = _ring_relative_contrast(smoothed, (row, col), radius, n_angles)
        if evidence is None:
            return -np.inf, []
        details.append({"hkl": hkl, "radius_px": float(radius), **evidence})
    contrasts = [d["contrast"] for d in details]
    # A single bright partial arc cannot compensate for an absent companion.
    score = min(contrasts) + 0.15 * sum(contrasts)
    return float(score), details


def _search_proposal(
    smoothed: np.ndarray,
    center_bounds: tuple[float, float, float, float],
    distance_bounds_mm: tuple[float, float],
    ratio_110_100: float = TAN_RATIO_110_100,
) -> dict:
    row0, row1, col0, col1 = center_bounds
    r0 = radius_from_distance(distance_bounds_mm[0])
    r1 = radius_from_distance(distance_bounds_mm[1])
    coarse = []
    for row in np.arange(np.floor(row0 / 16) * 16, row1 + 8, 16):
        for col in np.arange(np.floor(col0 / 16) * 16, col1 + 8, 16):
            for radius in np.arange(np.floor(r0 / 8) * 8, r1 + 4, 8):
                score, details = _pair_score(smoothed, row, col, radius, 180, ratio_110_100)
                if np.isfinite(score):
                    coarse.append((score, float(row), float(col), float(radius), details))
    if not coarse:
        raise RuntimeError("no two-ring proposal had sufficient detector coverage")
    coarse.sort(reverse=True, key=lambda item: item[0])
    _, best_row, best_col, best_radius, _ = coarse[0]

    fine = []
    for row in np.arange(best_row - 16, best_row + 17, 1):
        for col in np.arange(best_col - 16, best_col + 17, 1):
            for radius in np.arange(best_radius - 12, best_radius + 13, 1):
                score, details = _pair_score(smoothed, row, col, radius, 360, ratio_110_100)
                if np.isfinite(score):
                    fine.append((score, float(row), float(col), float(radius), details))
    fine.sort(reverse=True, key=lambda item: item[0])
    score, row, col, radius, details = fine[0]
    alternatives = [item for item in coarse if np.hypot(item[1] - row, item[2] - col) >= 48]
    alternative_score = alternatives[0][0] if alternatives else 0.0
    return {
        "center": [row, col],
        "radius_100_px": radius,
        "radius_110_px": radius * ratio_110_100,
        "score": score,
        "alternative_score": alternative_score,
        "pair_evidence": details,
        "center_bounds": list(center_bounds),
        "distance_bounds_mm": list(distance_bounds_mm),
    }


def _circular_span_deg(theta: np.ndarray) -> float:
    angles = np.sort(np.mod(theta, 2 * np.pi))
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def _extract_ridge_points(
    image: np.ndarray,
    center: tuple[float, float],
    radius: float,
    hkl: str,
) -> np.ndarray:
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    offsets = np.arange(-18.0, 18.01, 0.5)
    points = []
    for theta in theta_grid:
        rr = center[0] + (radius + offsets) * np.sin(theta)
        cc = center[1] + (radius + offsets) * np.cos(theta)
        values = ndimage.map_coordinates(image, [rr, cc], order=1, mode="constant", cval=np.nan)
        if np.isfinite(values).sum() < 50:
            continue
        fill = np.where(np.isfinite(values), values, np.nanmedian(values))
        smooth_line = ndimage.gaussian_filter1d(fill, 1.0)
        peak = int(np.argmax(smooth_line))
        if peak < 7 or peak > len(offsets) - 8:
            continue
        edge = np.r_[smooth_line[:14], smooth_line[-14:]]
        background = float(np.median(edge))
        noise = 1.4826 * float(np.median(np.abs(edge - background)))
        prominence = float(smooth_line[peak] - background)
        if prominence < max(12.0, 3.0 * noise):
            continue
        window = slice(peak - 3, peak + 4)
        weights = np.maximum(smooth_line[window] - background, 0.0)
        if weights.sum() == 0:
            continue
        offset = float(np.sum(offsets[window] * weights) / weights.sum())
        points.append(
            (
                center[0] + (radius + offset) * np.sin(theta),
                center[1] + (radius + offset) * np.cos(theta),
                0 if hkl == "100" else 1,
                theta,
                prominence,
            )
        )
    return np.asarray(points, dtype=float)


def _fit_single_circle(points: np.ndarray, initial: tuple[float, float, float]) -> dict:
    def residual(params):
        return np.hypot(points[:, 0] - params[0], points[:, 1] - params[1]) - params[2]

    fit = optimize.least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=1000)
    raw = residual(fit.x)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw)))
    return {
        "center": [float(fit.x[0]), float(fit.x[1])],
        "radius_px": float(fit.x[2]),
        "residual_mad_px": float(mad),
        "n_points": int(len(points)),
        "angular_span_deg": _circular_span_deg(points[:, 3]),
    }


def _fit_joint_circles(
    points_100: np.ndarray,
    points_110: np.ndarray,
    initial: tuple[float, float, float, float],
    theoretical_ratio: float = TAN_RATIO_110_100,
) -> dict:
    points = np.vstack([points_100, points_110])
    counts = [len(points_100), len(points_110)]

    def residual(params):
        radii = np.where(points[:, 2] == 0, params[2], params[3])
        raw = np.hypot(points[:, 0] - params[0], points[:, 1] - params[1]) - radii
        weights = np.where(
            points[:, 2] == 0,
            np.sqrt((counts[0] + counts[1]) / (2 * counts[0])),
            np.sqrt((counts[0] + counts[1]) / (2 * counts[1])),
        )
        return raw * weights

    fit = optimize.least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=2000)
    row, col, radius_100, radius_110 = map(float, fit.x)
    raw = np.hypot(points[:, 0] - row, points[:, 1] - col) - np.where(
        points[:, 2] == 0, radius_100, radius_110
    )
    per_ring = {}
    for index, hkl in enumerate(("100", "110")):
        ring_residual = raw[points[:, 2] == index]
        per_ring[hkl] = {
            "residual_mad_px": float(
                1.4826 * np.median(np.abs(ring_residual - np.median(ring_residual)))
            ),
            "residual_p95_px": float(np.percentile(np.abs(ring_residual), 95)),
        }
    observed_ratio = radius_110 / radius_100
    return {
        "center": [row, col],
        "radii_px": {"100": radius_100, "110": radius_110},
        "observed_ratio": observed_ratio,
        "theoretical_ratio": theoretical_ratio,
        "ratio_error_pct": (observed_ratio / theoretical_ratio - 1) * 100,
        "distance_eff_mm": {
            "100": distance_from_radius(radius_100, "100"),
            "110": distance_from_radius(radius_110, "110"),
        },
        "per_ring_residual": per_ring,
    }


def _verify_fit(
    independent: dict,
    joint: dict,
    proposal: dict,
    ring_validation: dict | None = None,
) -> dict:
    center_100 = np.asarray(independent["100"]["center"])
    center_110 = np.asarray(independent["110"]["center"])
    center_separation = float(np.linalg.norm(center_100 - center_110))
    min_span = min(independent[hkl]["angular_span_deg"] for hkl in ("100", "110"))
    min_contrast = min(d["contrast"] for d in proposal["pair_evidence"])
    evidence = {
        "independent_center_separation_px": center_separation,
        "minimum_angular_span_deg": min_span,
        "minimum_pair_contrast": min_contrast,
        "ratio_error_pct": joint["ratio_error_pct"],
        "joint_residual_mad_px": {
            hkl: joint["per_ring_residual"][hkl]["residual_mad_px"] for hkl in ("100", "110")
        },
    }
    if ring_validation is not None:
        evidence["all_ring_location_validation"] = {
            "provider": ring_validation["ring_model"]["provider"],
            "calibrant": ring_validation["ring_model"]["calibrant"],
            "n_in_field": ring_validation["n_in_field"],
            "n_matched": ring_validation["n_matched"],
            "failed_hkl": ring_validation["failed_hkl"],
            "all_in_field_match": ring_validation["all_in_field_match"],
        }
    if min_contrast <= 0:
        return {
            "verdict": "revise",
            "failure_type": "detection_failure",
            "message": "Both verified reflections did not provide positive ridge evidence.",
            "evidence": evidence,
        }
    if ring_validation is not None and not ring_validation["all_in_field_match"]:
        return {
            "verdict": "revise",
            "failure_type": "ring_semantics_failure",
            "message": (
                "The proposed ring order is inconsistent with the independent "
                "pyFAI all-ring location check."
            ),
            "evidence": evidence,
        }
    if abs(joint["ratio_error_pct"]) > 1.0:
        return {
            "verdict": "escalate",
            "failure_type": "model_contradiction",
            "message": "Observed free-radius ratio is inconsistent with LaB6 (100)/(110).",
            "evidence": evidence,
        }
    # A short partial arc makes its independent circle center intrinsically
    # unstable. Keep angular span as uncertainty evidence, but do not turn that
    # limitation (or the resulting small center drift) into a terminal decision.
    # Independent-center disagreement is a hard contradiction only when both
    # rings have enough angular support for that comparison to be meaningful.
    if center_separation > 8.0 and min_span >= 25.0:
        return {
            "verdict": "escalate",
            "failure_type": "identifiability_failure",
            "message": "Well-covered rings independently imply different centers.",
            "evidence": evidence,
        }
    if max(evidence["joint_residual_mad_px"].values()) > 3.0:
        return {
            "verdict": "revise",
            "failure_type": "fit_failure",
            "message": "Common-center residuals are too large; revise ridge extraction.",
            "evidence": evidence,
        }
    return {
        "verdict": "accept",
        "failure_type": None,
        "message": (
            "The pyFAI ring semantics and joint circle fit are consistent; short "
            "angular coverage is retained as uncertainty evidence only."
        ),
        "evidence": evidence,
    }


def estimate_center(
    img: np.ndarray,
    center_bounds: tuple[float, float, float, float] | None = None,
    distance_bounds_mm: tuple[float, float] = DEFAULT_DISTANCE_BOUNDS_MM,
) -> tuple[dict, dict[str, np.ndarray]]:
    height, width = img.shape
    if center_bounds is None:
        # Instrument-context prior from the lower-left beamstop region; it
        # contains no expected ring radius and remains deliberately broad.
        center_bounds = (0.82 * height, 1.05 * height, -0.06 * width, 0.18 * width)
    ring_model = load_pyfai_lab6_model(wavelength_A=LAMBDA_A)
    ratio_110_100 = ring_model["rings"][1]["radius_ratio_to_100"]
    search_image = _smoothed_image(img, sigma=2.0)
    ridge_image = _smoothed_image(img, sigma=1.2)
    proposal = _search_proposal(
        search_image,
        center_bounds,
        distance_bounds_mm,
        ratio_110_100,
    )
    proposal_center = tuple(proposal["center"])
    points_100 = _extract_ridge_points(
        ridge_image, proposal_center, proposal["radius_100_px"], "100"
    )
    points_110 = _extract_ridge_points(
        ridge_image, proposal_center, proposal["radius_110_px"], "110"
    )
    if len(points_100) < 40 or len(points_110) < 25:
        verification = {
            "verdict": "revise",
            "failure_type": "detection_failure",
            "message": "Too few ridge points survived for a two-ring fit.",
            "evidence": {"n_points": {"100": len(points_100), "110": len(points_110)}},
        }
        result = {
            "ring_model": ring_model,
            "proposal": proposal,
            "independent_fits": {},
            "joint_fit": None,
            "ring_validation": None,
            "verification": verification,
        }
        return result, {"100": points_100, "110": points_110}

    independent = {
        "100": _fit_single_circle(points_100, (*proposal_center, proposal["radius_100_px"])),
        "110": _fit_single_circle(points_110, (*proposal_center, proposal["radius_110_px"])),
    }
    joint = _fit_joint_circles(
        points_100,
        points_110,
        (*proposal_center, proposal["radius_100_px"], proposal["radius_110_px"]),
        theoretical_ratio=ratio_110_100,
    )
    ring_validation = validate_all_predicted_rings(
        img,
        tuple(joint["center"]),
        joint["radii_px"]["100"],
        ring_model=ring_model,
    )
    result = {
        "ring_model": ring_model,
        "proposal": proposal,
        "independent_fits": independent,
        "joint_fit": joint,
        "ring_validation": ring_validation,
        "verification": _verify_fit(
            independent,
            joint,
            proposal,
            ring_validation,
        ),
    }
    return result, {"100": points_100, "110": points_110}


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


def _write_overlay(
    img: np.ndarray,
    result: dict,
    points: dict[str, np.ndarray],
    destination: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    valid = np.isfinite(img) & (img != 0)
    vmin, vmax = np.percentile(img[valid], [2.0, 99.5])
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    joint = result.get("joint_fit")
    if joint:
        row, col = joint["center"]
        verdict = result["verification"]["verdict"]
        ax.scatter(
            [col],
            [row],
            marker="+",
            s=180,
            linewidths=2.2,
            color="#ff3b30",
            label=f"candidate center ({row:.1f}, {col:.1f}); {verdict}",
        )
        colors = {"100": "#ffffff", "110": "#ffcc00"}
        for hkl in ("100", "110"):
            radius = joint["radii_px"][hkl]
            ax.add_patch(
                Circle(
                    (col, row),
                    radius,
                    fill=False,
                    lw=1.2,
                    color=colors[hkl],
                    label=f"free-fit LaB6 ({hkl}), r={radius:.1f}px",
                )
            )
            step = max(1, len(points[hkl]) // 120)
            ring_points = points[hkl][::step]
            if len(ring_points):
                ax.scatter(ring_points[:, 1], ring_points[:, 0], s=5, color=colors[hkl], alpha=0.55)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_xlabel("assembled column (px)")
    ax.set_ylabel("assembled row (px)")
    ax.set_title("Image-center hypothesis: verified lattice rings only")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def _write_human_review(out: Path, result: dict) -> None:
    verification = result["verification"]
    joint = result.get("joint_fit") or {}
    lines = [
        "# Image-center escalation packet",
        "",
        f"- Verdict: **{verification['verdict']}**",
        f"- Failure type: `{verification.get('failure_type')}`",
        f"- Reason: {verification['message']}",
        "- Only LaB6 (100)/(110) were used; 367/531/734 px were excluded.",
    ]
    if joint:
        row, col = joint["center"]
        lines.extend(
            [
                f"- Joint candidate center: row {row:.2f}, col {col:.2f}",
                f"- Free radii: (100) {joint['radii_px']['100']:.2f} px; "
                f"(110) {joint['radii_px']['110']:.2f} px",
                f"- Radius-ratio error: {joint['ratio_error_pct']:.3f}%",
                f"- Effective distances: (100) {joint['distance_eff_mm']['100']:.2f} mm; "
                f"(110) {joint['distance_eff_mm']['110']:.2f} mm",
            ]
        )
    independent = result.get("independent_fits") or {}
    for hkl in ("100", "110"):
        fit = independent.get(hkl)
        if fit:
            lines.append(
                f"- Independent ({hkl}) center: row {fit['center'][0]:.2f}, "
                f"col {fit['center'][1]:.2f}; arc span {fit['angular_span_deg']:.2f}°; "
                f"residual MAD {fit['residual_mad_px']:.2f} px"
            )
    evidence = verification.get("evidence") or {}
    if evidence:
        lines.append(
            "- Independent-center separation: "
            f"{evidence.get('independent_center_separation_px', float('nan')):.2f} px"
        )
    ring_validation = result.get("ring_validation")
    if ring_validation:
        lines.append(
            "- Independent pyFAI all-ring validation: "
            f"{ring_validation['n_matched']}/{ring_validation['n_in_field']} matched; "
            f"failed orders: {ring_validation['failed_hkl'] or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Researcher decision requested",
            "",
            "Choose whether to accept the joint candidate using detector-geometry context, "
            "provide a previously measured center, or authorize a tilted/elliptical model.",
        ]
    )
    (out / "center_human_review.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="agent_estimate")
    ap.add_argument(
        "--center-bounds", nargs=4, type=float, metavar=("ROW_MIN", "ROW_MAX", "COL_MIN", "COL_MAX")
    )
    ap.add_argument(
        "--distance-bounds-mm",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=DEFAULT_DISTANCE_BOUNDS_MM,
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    img = np.load(out / "sum_assembled.npy")
    result, points = estimate_center(
        img,
        tuple(args.center_bounds) if args.center_bounds else None,
        tuple(args.distance_bounds_mm),
    )
    verification = result["verification"]
    joint = result.get("joint_fit")
    ring_model = result["ring_model"]
    ring_validation = result.get("ring_validation")
    accepted = verification["verdict"] == "accept"
    artifact = {
        "schema_version": 2,
        "coordinate_system": "assembled_row_col",
        "decision": "estimate" if accepted else verification["verdict"],
        "activated": True,
        "specified_in_context": False,
        "center": (
            {"row": joint["center"][0], "col": joint["center"][1]} if accepted and joint else None
        ),
        "candidate_center": (
            {"row": joint["center"][0], "col": joint["center"][1]} if joint else None
        ),
        "source": args.source,
        "method": (
            "free-radius two-ring fit using pyFAI LaB6 order relationships, "
            "followed by independent all-ring pixel validation"
        ),
        "confidence": "accepted" if accepted else "not_accepted",
        "lattice_model": {
            "material": "LaB6",
            "lattice_constant_A": LATTICE_A_A,
            "wavelength_A": LAMBDA_A,
            "pixel_mm": PIXEL_MM,
            "verified_reflections": ["100", "110"],
            "relationship_provider": ring_model["provider"],
            "pyfai_version": ring_model["pyfai_version"],
            "calibrant": ring_model["calibrant"],
            "theoretical_radius_ratio_110_100": (ring_model["rings"][1]["radius_ratio_to_100"]),
            "excluded_unindexed_anchors_px": UNINDEXED_ANCHORS_PX,
        },
        **result,
    }
    artifact = _round_nested(artifact)
    (out / "image_center.json").write_text(json.dumps(artifact, indent=2) + "\n")
    if ring_validation:
        rounded_validation = _round_nested(ring_validation)
        (out / "ring_validation.json").write_text(json.dumps(rounded_validation, indent=2) + "\n")
        write_validation_overlay(
            img,
            rounded_validation,
            out / "ring_validation_overlay.png",
        )
    loop_path = out / "center_loop.jsonl"
    attempts = len(loop_path.read_text().splitlines()) if loop_path.exists() else 0
    with loop_path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "attempt": attempts + 1,
                    "source": args.source,
                    "decision": artifact["decision"],
                    "candidate_center": artifact.get("candidate_center"),
                    "all_ring_location_validation": (
                        artifact["verification"]["evidence"].get("all_ring_location_validation")
                    ),
                    "verification": artifact["verification"],
                }
            )
            + "\n"
        )
    _write_overlay(img, result, points, out / "image_center_overlay.png")
    review_path = out / "center_human_review.md"
    if verification["verdict"] == "escalate":
        _write_human_review(out, result)
    elif review_path.exists():
        # Do not leave a stale escalation packet beside a newly accepted result.
        review_path.unlink()
    print(
        f"[center] verdict={verification['verdict']} "
        f"failure_type={verification.get('failure_type')} -> {out / 'image_center.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
