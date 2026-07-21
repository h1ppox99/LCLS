#!/usr/bin/env python3
"""
sweep_hyperparameters.py -- the one general sweep driver (Hydra).

Builds a masking.Pipeline from the selected stat / regularization / mask_reg /
combine config groups (or a full `detectors` list from an experiment), evaluates
it across `eval.runs`, prints a metric row, and appends it to a results.csv in the
Hydra output dir. Every specialized scripts/*.sh is a thin wrapper over this file.

Single detector, default groups:
    python studies/sweep_hyperparameters.py

Sweep (multirun): one job per (stat, reg, combine, params) combo:
    python studies/sweep_hyperparameters.py -m \
        stat=variance regularization=tv combine=union \
        stat.params.k=2,2.5,3,3.5 regularization.params.weight=5,10,15

Full production recipe (regression anchor):
    python studies/sweep_hyperparameters.py experiment=production
    python studies/sweep_hyperparameters.py experiment=production combine=weighted_sum
"""
from __future__ import annotations
import os, sys, csv

import hydra
from omegaconf import DictConfig, OmegaConf

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOMASK)
from masking import Detector, Pipeline, STATS, REGULARIZERS, COMBINERS  # noqa: E402
from evaluation import evaluate                                          # noqa: E402


def _params(registry, name, cfg_params):
    """Instantiate a spec's Params dataclass from an OmegaConf params node."""
    kwargs = OmegaConf.to_container(cfg_params, resolve=True) if cfg_params else {}
    return registry[name].params(**kwargs)


def _detector_from_dict(d: dict) -> Detector:
    """Build a Detector from a plain dict (an entry of an experiment `detectors`)."""
    sp = STATS[d["stat"]].params(**(d.get("stat_params") or {}))
    freg = d.get("field_reg")
    mreg = d.get("mask_reg")
    return Detector(
        d["stat"], sp,
        field_reg=freg,
        field_reg_params=REGULARIZERS[freg].params(**(d.get("field_reg_params") or {})) if freg else None,
        mask_reg=mreg,
        mask_reg_params=REGULARIZERS[mreg].params(**(d.get("mask_reg_params") or {})) if mreg else None,
    )


def build_pipeline(cfg: DictConfig) -> Pipeline:
    combiner = cfg.combine.name
    cparams = _params(COMBINERS, combiner, cfg.combine.get("params"))

    if cfg.get("detectors"):
        detectors = [_detector_from_dict(OmegaConf.to_container(d, resolve=True))
                     for d in cfg.detectors]
    else:
        freg = cfg.regularization.name
        mreg = cfg.mask_reg.name
        detectors = [Detector(
            cfg.stat.name, _params(STATS, cfg.stat.name, cfg.stat.get("params")),
            field_reg=freg,
            field_reg_params=_params(REGULARIZERS, freg, cfg.regularization.get("params")) if freg else None,
            mask_reg=mreg,
            mask_reg_params=_params(REGULARIZERS, mreg, cfg.mask_reg.get("params")) if mreg else None,
        )]
    return Pipeline(detectors, combiner=combiner, combiner_params=cparams)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pipe = build_pipeline(cfg)
    runs = list(cfg.eval.runs)
    label = (",".join(d.stat for d in pipe.detectors) + f" | {cfg.combine.name}")
    print(f"=== {label}  (runs {runs}) ===")
    metrics = evaluate(pipe, runs, verbose=True)
    mean = metrics["mean"]
    print(f"  MEAN: IoU {mean['iou']:.3f}  prec {mean['precision']:.3f}  "
          f"rec {mean['recall']:.3f}  (residual IoU {mean['residual_iou']:.3f})")

    # Flatten the swept overrides for the results row.
    row = {"detectors": label, "combine": cfg.combine.name,
           "mean_iou": mean["iou"], "mean_precision": mean["precision"],
           "mean_recall": mean["recall"], "mean_residual_iou": mean["residual_iou"]}
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
    csv_path = os.path.join(base, "results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)

    if cfg.figures:
        import viz
        from evaluation import load_sample
        fig_dir = os.path.join(AUTOMASK, "outputs", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for run in runs:
            s = load_sample(run)
            out = os.path.join(fig_dir, f"sweep_{cfg.stat.name}_run{run:04d}.png")
            viz.save_agreement(pipe.run(s), pipe.floor(s), s.human, run, out,
                               title=f"{label} — run {run}")
            print(f"  [saved] {out}")

    return mean["iou"]


if __name__ == "__main__":
    main()
