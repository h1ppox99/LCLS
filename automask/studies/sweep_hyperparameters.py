#!/usr/bin/env python3
"""Hydra sweep driver for the recovered masking pipeline."""
from __future__ import annotations

import csv
import os

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from automask.combine.base import COMBINERS
from automask.evaluation import FIT_RUNS, VALIDATION_RUNS, evaluate, reference_mask
from automask.masking import Channel, Pipeline, floor_channels
from automask.sample import Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.regularization.base import REGULARIZERS
from automask.stats.base import STATS

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_plain(value):
    """Convert an OmegaConf node to a plain dict/list; pass plain Python objects through
    (needed because `_detector_from_dict` already converted its input in one shot)."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _params(registry, name, cfg_params):
    kwargs = _to_plain(cfg_params) if cfg_params else {}
    return registry[name].params(**kwargs)


def _reg_params(names, cfg_params):
    """Build regularizer params for a slot that may name one stage or several.

    Mirrors `Channel._stages`: `names` is None / a name / a list of names, and
    `cfg_params` is correspondingly None / a mapping / a list of mappings. Returns
    the params in the shape `Channel` expects."""
    if not names:
        return None
    raw = _to_plain(cfg_params) if cfg_params else None
    if isinstance(names, str):
        return REGULARIZERS[names].params(**(raw or {}))
    names = list(names)
    if raw is None:
        raw = [None] * len(names)
    if len(raw) != len(names):
        raise ValueError(f"regularizer list {names} has {len(names)} entries but "
                         f"{len(raw)} params entries")
    return [REGULARIZERS[n].params(**(r or {})) for n, r in zip(names, raw)]


def _reg_names(value):
    """Normalize a conf `name:` field that may be a string or a list."""
    if value is None or isinstance(value, str):
        return value
    return list(_to_plain(value))


def _channel_from_dict(channel: dict) -> Channel:
    stat = channel["stat"]
    field_reg = _reg_names(channel.get("field_reg"))
    mask_reg = _reg_names(channel.get("mask_reg"))
    return Channel(
        stat, STATS[stat].params(**(channel.get("params") or {})),
        field_reg=field_reg,
        field_reg_params=_reg_params(field_reg, channel.get("field_reg_params")),
        mask_reg=mask_reg,
        mask_reg_params=_reg_params(mask_reg, channel.get("mask_reg_params")),
        name=channel.get("name"),
    )


def build_pipeline(cfg: DictConfig) -> Pipeline:
    """The swept evidence channels on the standard intensity-free floor.

    The conf lists evidence channels only; the floor is always the same and is
    not swept, so it is added here rather than repeated in every experiment file.
    """
    combiner = cfg.combine.name
    combiner_params = _params(COMBINERS, combiner, cfg.combine.get("params"))
    if cfg.get("channels"):
        evidence = [_channel_from_dict(OmegaConf.to_container(c, resolve=True))
                    for c in cfg.channels]
    else:
        field_reg = _reg_names(cfg.regularization.name)
        mask_reg = _reg_names(cfg.mask_reg.name)
        evidence = [Channel(
            cfg.stat.name, _params(STATS, cfg.stat.name, cfg.stat.get("params")),
            field_reg=field_reg,
            field_reg_params=_reg_params(field_reg, cfg.regularization.get("params")),
            mask_reg=mask_reg,
            mask_reg_params=_reg_params(mask_reg, cfg.mask_reg.get("params")),
        )]
    return Pipeline(floor_channels() + evidence, combiner=combiner,
                    combiner_params=combiner_params)


def _runs_for_phase(eval_cfg):
    phase = str(eval_cfg.phase)
    if phase == "fit":
        return phase, list(eval_cfg.fit_runs or FIT_RUNS)
    if phase == "validate":
        return phase, list(eval_cfg.validation_runs or VALIDATION_RUNS)
    raise ValueError("eval.phase must be 'fit' or 'validate'")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pipe = build_pipeline(cfg)
    phase, runs = _runs_for_phase(cfg.eval)
    label = (",".join(c.label for c in pipe.evidence_channels)
             + f" | {cfg.combine.name}")
    print(f"=== {phase}: {label}  (runs {runs}) ===")

    metrics = evaluate(pipe, runs, verbose=True)
    mean = metrics["mean"]
    print(f"  MEAN: IoU {mean['iou']:.3f}  prec {mean['precision']:.3f}  "
          f"rec {mean['recall']:.3f}  (residual IoU {mean['residual_iou']:.3f})")

    # Flatten the swept overrides for the results row.
    row = {"phase": phase, "runs": ",".join(str(run) for run in runs),
           "channels": label, "combine": cfg.combine.name,
           "mean_iou": mean["iou"], "mean_precision": mean["precision"],
           "mean_recall": mean["recall"],
           "mean_residual_iou": mean["residual_iou"]}
    for k in ("stat", "regularization", "mask_reg"):
        p = cfg.get(k, {}).get("params") if cfg.get(k) else None
        if p:
            for pk, pv in OmegaConf.to_container(p, resolve=True).items():
                row[f"{k}.{pk}"] = pv
    cp = cfg.combine.get("params")
    if cp:
        for pk, pv in OmegaConf.to_container(cp, resolve=True).items():
            row[f"combine.{pk}"] = pv

    # Append into the multirun sweep dir when sweeping, else the run dir. Both are
    # relative to Hydra's ORIGINAL cwd, but jobs chdir into their per-job subdir,
    # so resolve to an absolute path first.
    try:
        from hydra.core.hydra_config import HydraConfig
        hc = HydraConfig.get()
        base = hc.sweep.dir if hc.mode.name == "MULTIRUN" else hc.run.dir
        if not os.path.isabs(base):
            base = os.path.join(hc.runtime.cwd, base)
    except Exception:
        base = os.getcwd()
    os.makedirs(base, exist_ok=True)
    csv_path = os.path.join(base, "results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [saved] {csv_path}")

    if cfg.figures:
        from automask import viz
        figure_dir = os.path.join(AUTOMASK, "outputs", "figures")
        os.makedirs(figure_dir, exist_ok=True)
        for run in runs:
            sample = Sample.from_store(run, BEAM_ON_SELECTION, pipe.needs())
            out = os.path.join(figure_dir, f"sweep_{cfg.stat.name}_run{run:04d}.png")
            viz.save_agreement(pipe.run(sample), pipe.floor(sample),
                               reference_mask(run), run, out,
                               title=f"{label} — run {run}")
            print(f"  [saved] {out}")
            panels = os.path.join(figure_dir, f"panels_{cfg.stat.name}_run{run:04d}.png")
            viz.channel_panels(pipe, sample, out=panels)
            print(f"  [saved] {panels}")
    return mean["residual_iou"]


if __name__ == "__main__":
    main()
