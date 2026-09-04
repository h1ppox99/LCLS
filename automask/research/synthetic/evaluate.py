#!/usr/bin/env python3
"""
synthetic/evaluate.py -- run the synthetic-artifact evaluation from a YAML config.

For each artifact type it injects ``n_per_artifact`` deterministic examples into
the valid pixels of one source image, runs the masker on each corrupted image,
scores the prediction against the injected mask over the originally-valid region,
appends a row to ``results.csv`` and writes one diagnostic figure per example.

It additionally evaluates an ``original`` case per source/rotation: nothing is
injected and the masker is scored against the REAL ground-truth mask over the
whole frame, so the suite also measures recovery of the original (un-corrupted)
mask, not only the injected artifacts.

    python -m automask.research.synthetic.evaluate --config automask/synthetic/config/synthetic_baseline.yaml

Config keys (see the shipped synthetic_baseline.yaml):
    image / mask   a run number (its cached mean / packaged reference mask) OR a .npy path
    masker         "module:function" single-image masker (default mask_image)
    output_dir     where results.csv + figures/ are written
    seed           base seed; per-example seeds are derived deterministically
    n_per_artifact examples generated per artifact type
    rotations      list of 90-deg multiples; image + ground-truth are rotated
                   together (np.rot90) and each orientation is evaluated
    artifacts      per-type parameter dict; a [lo, hi] value is sampled uniformly
                   per example (int stays int); 'center' is passed through as-is
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from importlib import import_module

import numpy as np

import matplotlib

if not os.environ.get("MPLBACKEND") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automask.research.synthetic.artifacts import ARTIFACTS
from automask.research.synthetic.metrics import masking_metrics
from automask.viz import agree_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config", "synthetic_baseline.yaml")
_METRIC_COLS = ["precision", "recall", "f1", "iou", "fpr", "masked_frac"]

# The no-artifact case: inject nothing and score the masker against the REAL
# ground-truth mask over the whole frame. Evaluated alongside the injected
# artifacts so the synthetic suite also measures recovery of the original mask.
ORIGINAL = "original"


def _cases():
    """Evaluated case order: every injected artifact, then the original-mask case.
    Appending ORIGINAL last keeps the per-example seeds of the artifacts stable."""
    return list(ARTIFACTS) + [ORIGINAL]


# --------------------------------------------------------------------------
# loading helpers -- accept a filesystem .npy path, a run number, or a mask name
# --------------------------------------------------------------------------
def _is_path(spec: str) -> bool:
    return spec.endswith(".npy") or os.sep in spec


def _load_source_image(spec) -> tuple[np.ndarray, str]:
    """A `.npy` path, or a run number whose selected-shot mean to corrupt."""
    if isinstance(spec, str) and _is_path(spec):
        return np.load(spec).astype(np.float64), os.path.splitext(
            os.path.basename(spec)
        )[0]
    try:
        run = int(spec)
    except (TypeError, ValueError):
        raise ValueError(
            f"image source {spec!r} is neither a .npy path nor a run number"
        ) from None
    from automask.sample.image_store import ImageStore
    from automask.selection.presets import BEAM_ON_SELECTION

    image = ImageStore().reduce(run, BEAM_ON_SELECTION, "mean").astype(np.float64)
    return image, f"run{run:04d}"


def _load_gt_mask(spec) -> np.ndarray:
    """A `.npy` path, or a run number whose packaged reference mask to load."""
    if isinstance(spec, str) and _is_path(spec):
        return np.load(spec).astype(bool)
    from automask.evaluation import reference_mask

    try:
        run = int(spec)
    except (TypeError, ValueError):
        raise ValueError(
            f"mask source {spec!r} is neither a .npy path nor a run number"
        ) from None
    return reference_mask(run)


def _resolve_masker(spec: str):
    """ "module:function" -> callable (the project's masker convention)."""
    if ":" not in spec:
        raise ValueError(f"masker must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    func = getattr(import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"{spec} is not callable")
    return func


def _rotate(image: np.ndarray, gt: np.ndarray, degrees: int):
    """Rotate an image and its ground-truth mask together by a multiple of 90 deg.

    Rotating both with the same np.rot90 keeps them aligned (90/270 also swap the
    two axes, giving a transposed shape -- fine, the masker takes any 2-D array).
    """
    if int(degrees) % 90 != 0:
        raise ValueError(f"rotations must be multiples of 90, got {degrees}")
    k = (int(degrees) // 90) % 4
    return np.rot90(image, k), np.rot90(gt, k)


# --------------------------------------------------------------------------
# parameter sampling -- resolve [lo, hi] ranges to a concrete value per example
# --------------------------------------------------------------------------
def _sample_params(rng, params: dict | None) -> dict:
    out = {}
    for key, val in (params or {}).items():
        is_range = (
            key != "center"
            and isinstance(val, (list, tuple))
            and len(val) == 2
            and all(isinstance(x, (int, float)) for x in val)
        )
        if is_range:
            lo, hi = val
            sampled = rng.uniform(lo, hi)
            out[key] = (
                int(round(sampled))
                if isinstance(lo, int) and isinstance(hi, int)
                else float(sampled)
            )
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------
# display transform (figures ONLY -- never applied to evaluated data)
# --------------------------------------------------------------------------
def _display(img: np.ndarray) -> np.ndarray:
    """Robust percentile clip + asinh, for imshow only."""
    lo, hi = np.percentile(img, [1.0, 99.0])
    return np.arcsinh(np.clip(img, lo, hi) - lo)


def _figure(original, corrupted, injected, pred, region, title, path):
    """Five-panel diagnostic: original, corrupted, injected, predicted, TP/FP/FN."""
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    ax[0].imshow(_display(original), cmap="magma")
    ax[0].set_title("original")
    ax[1].imshow(_display(corrupted), cmap="magma")
    ax[1].set_title("corrupted")
    ax[2].imshow(injected, cmap="gray")
    ax[2].set_title("injected mask")
    ax[3].imshow(pred, cmap="gray")
    ax[3].set_title("predicted mask")
    ax[4].imshow(agree_rgb(pred & region, injected & region))
    ax[4].set_title("overlay (green=TP red=FP blue=FN)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# per-mode context -- each yields, for a rotation, an "inject + predict" closure
# returning (original_img, corrupted_img, injected, pred, region)
# --------------------------------------------------------------------------
def _image_context(cfg):
    """mode="image": corrupt one 2-D image, run a module:function masker."""
    image, image_id = _load_source_image(cfg["image"])
    gt = _load_gt_mask(cfg["mask"])
    if gt.shape != image.shape:
        raise ValueError(f"mask shape {gt.shape} != image shape {image.shape}")
    masker_spec = cfg.get("masker", "automask.mask:mask_image")
    masker = _resolve_masker(masker_spec)

    def for_rotation(degrees):
        rimage, rgt = _rotate(image, gt, degrees)  # image + mask rotate together
        region = ~rgt
        full = np.ones_like(rgt, dtype=bool)

        def run_example(name, rng, params):
            if name == ORIGINAL:  # no injection: score vs the real mask
                pred = np.asarray(masker(rimage)).astype(bool)
                return rimage, rimage, rgt, pred, full
            corrupted, injected = ARTIFACTS[name](rimage, region, rng, **params)
            pred = np.asarray(masker(corrupted)).astype(bool)
            return rimage, corrupted, injected, pred, region

        return run_example

    return [(image_id, for_rotation)], masker_spec


def _pipeline_context(cfg):
    """mode="pipeline": corrupt a full Sample, run the production Pipeline.

    Supports one run (`run:`), several (`runs: [389, 475]`), or every local XTC
    run (`runs: null`) -- each becomes a source with the same artifact suite.
    """
    from automask.mask import production_pipeline
    from automask.evaluation import reference_mask
    from automask.sample import Sample
    from automask.selection.presets import BEAM_ON_SELECTION
    from automask.research.synthetic.sample_adapter import corrupt_sample, rotate_sample

    from automask.evaluation import ALL_RUNS

    if "runs" in cfg:
        runs = list(ALL_RUNS if cfg["runs"] is None else cfg["runs"])
    else:
        runs = [int(cfg["run"])]
    runs = [int(run) for run in runs]
    pipe = cfg.get("_pipeline")  # a swept Pipeline, if provided
    if pipe is None:
        pipe = production_pipeline()
        model = "production_pipeline()"
    else:
        model = (
            "swept_pipeline(" + ",".join(c.label for c in pipe.evidence_channels) + ")"
        )
    signature = (
        f"{model} "
        f"run{'s' if len(runs) > 1 else ''} {runs if len(runs) > 1 else runs[0]}"
    )

    def make_source(run):
        sample = Sample.from_store(run, BEAM_ON_SELECTION, pipe.needs())
        human = reference_mask(run)

        def for_rotation(degrees):
            k = (int(degrees) // 90) % 4
            rs = rotate_sample(sample, degrees)
            rhuman = np.rot90(human, k)  # the reference rotates with it
            region = ~rhuman  # originally-valid pixels
            full = np.ones_like(rhuman, dtype=bool)

            def run_example(name, rng, params):
                if name == ORIGINAL:  # no injection: score vs the real mask
                    pred = pipe.run(rs).astype(bool)
                    return rs.mean, rs.mean, rhuman, pred, full
                csample, injected = corrupt_sample(rs, name, rng, params, region)
                pred = pipe.run(csample).astype(bool)
                return rs.mean, csample.mean, injected, pred, region

            return run_example

        return f"run{run:04d}", for_rotation

    return [make_source(r) for r in runs], signature


_CONTEXTS = {"image": _image_context, "pipeline": _pipeline_context}


# --------------------------------------------------------------------------
# main evaluation loop
# --------------------------------------------------------------------------
def evaluate_config(cfg: dict, pipeline=None) -> dict:
    if pipeline is not None:  # score a caller-supplied Pipeline
        cfg = {**cfg, "_pipeline": pipeline}
    mode = cfg.get("mode", "image")
    if mode not in _CONTEXTS:
        raise ValueError(f"mode must be one of {sorted(_CONTEXTS)}, got {mode!r}")
    sources, signature = _CONTEXTS[mode](cfg)

    base_seed = int(cfg.get("seed", 0))
    n_per = int(cfg.get("n_per_artifact", 3))
    artifact_cfg = cfg.get("artifacts", {}) or {}
    rotations = cfg.get("rotations", [0, 90, 180, 270])
    save_figures = bool(cfg.get("save_figures", True))
    cases = _cases()

    out_dir = cfg.get("output_dir", os.path.join(HERE, "..", "outputs", "synthetic"))
    out_dir = os.path.abspath(out_dir)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    header = [
        "image_id",
        "rotation",
        "artifact_type",
        "seed",
        "injected_px",
        "params",
    ] + _METRIC_COLS
    rows = []
    for src_idx, (image_id, for_rotation) in enumerate(sources):
        for rot_idx, degrees in enumerate(rotations):
            run_example = for_rotation(degrees)
            for type_idx, name in enumerate(cases):
                # An artifact may restrict which rotations it is valid under (e.g.
                # `hot_patch` corrupts a panel-form constant, which has no image
                # rotation -- see sample_adapter.rotate_sample). Popped here so it
                # never reaches the generator as a parameter.
                acfg = dict(artifact_cfg.get(name) or {})
                allowed = acfg.pop("rotations", None)
                if allowed is not None and int(degrees) not in [
                    int(a) for a in allowed
                ]:
                    continue
                # the original-mask case injects nothing and is deterministic -> run once.
                n = 1 if name == ORIGINAL else n_per
                for i in range(n):
                    # deterministic per-example seed: same config -> same corruption
                    seed = (
                        base_seed
                        + src_idx * 1_000_000
                        + rot_idx * 100_000
                        + type_idx * 10_000
                        + i
                    )
                    rng = np.random.default_rng(seed)
                    params = _sample_params(rng, acfg)

                    original, corrupted, injected, pred, region = run_example(
                        name, rng, params
                    )
                    metrics = masking_metrics(pred, injected, region)

                    tag = f"{image_id}__rot{int(degrees):03d}__{name}__{i:02d}"
                    if save_figures:
                        _figure(
                            original,
                            corrupted,
                            injected,
                            pred,
                            region,
                            title=f"{tag}  seed={seed}  "
                            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                            f"F1={metrics['f1']:.2f} IoU={metrics['iou']:.2f}",
                            path=os.path.join(fig_dir, tag + ".png"),
                        )

                    rows.append(
                        {
                            "image_id": image_id,
                            "rotation": int(degrees),
                            "artifact_type": name,
                            "seed": seed,
                            "injected_px": int(injected.sum()),
                            "params": json.dumps(params),
                            **{k: round(metrics[k], 6) for k in _METRIC_COLS},
                        }
                    )
                    print(
                        f"{tag:44s} seed={seed:<7d} inj={int(injected.sum()):>7d} "
                        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                        f"F1={metrics['f1']:.3f} IoU={metrics['iou']:.3f} "
                        f"FPR={metrics['fpr']:.4f}"
                    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact_type", "n"] + _METRIC_COLS)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nwrote {len(rows)} rows -> {csv_path}")
    print(f"figures -> {fig_dir}")
    _print_summary(summary, signature)
    print(f"summary -> {summary_path}")
    return {"csv": csv_path, "summary_csv": summary_path, "summary": summary}


# --------------------------------------------------------------------------
# aggregation -- turn the per-example rows into a comparable evaluation signal
# --------------------------------------------------------------------------
def _summarize(rows: list[dict]) -> list[dict]:
    """Mean of every metric per artifact type (across rotations + examples),
    plus an 'overall' row. This is the aggregate score used to rank maskers."""

    def agg(name, rs):
        d = {"artifact_type": name, "n": len(rs)}
        d.update(
            {k: round(float(np.mean([r[k] for r in rs])), 6) for k in _METRIC_COLS}
        )
        return d

    groups = defaultdict(list)
    for r in rows:
        groups[r["artifact_type"]].append(r)
    summary = [agg(name, groups[name]) for name in _cases() if name in groups]
    if rows:
        summary.append(agg("overall", rows))
    return summary


def _print_summary(summary: list[dict], masker_spec: str) -> None:
    print(f"\n=== synthetic summary (masker={masker_spec}) ===")
    print(f"{'artifact':12s} {'n':>3s} " + " ".join(f"{k:>11s}" for k in _METRIC_COLS))
    for row in summary:
        print(
            f"{row['artifact_type']:12s} {row['n']:>3d} "
            + " ".join(f"{row[k]:11.4f}" for k in _METRIC_COLS)
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="path to the YAML config (default: shipped synthetic_baseline.yaml)",
    )
    args = ap.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate_config(cfg)


if __name__ == "__main__":
    main()
