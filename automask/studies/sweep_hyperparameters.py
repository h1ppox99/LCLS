def _synthetic_scores(cfg: DictConfig, pipe: Pipeline, runs, out_dir: str) -> dict:
    """Score `pipe` with the synthetic-artifact evaluation. Returns a metric dict
    exposing the same iou/precision/recall keys the sweep row expects, plus a
    per-case `iou_<case>` for every artifact and the `original` mask case."""
    from automask.synthetic.evaluate import evaluate_config

    ecfg = cfg.eval
    scfg_path = ecfg.get("synthetic_config", "synthetic/config/synthetic_pipeline.yaml")
    if not os.path.isabs(scfg_path):
        scfg_path = os.path.join(AUTOMASK, scfg_path)
    with open(scfg_path) as f:
        base = yaml.safe_load(f)

    scfg = {
        "mode": "pipeline",
        "runs": [int(r) for r in runs],
        "seed": int(ecfg.get("synthetic_seed", 0)),
        "n_per_artifact": int(ecfg.get("synthetic_n_per", 3)),
        "rotations": list(ecfg.get("synthetic_rotations", [0, 180])),
        "artifacts": base.get("artifacts", {}),
        "output_dir": os.path.join(out_dir, "synthetic"),
        "save_figures": False,           # keep sweeps fast: no per-example figures
    }
    result = evaluate_config(scfg, pipeline=pipe)
    by_case = {r["artifact_type"]: r for r in result["summary"]}
    overall = by_case["overall"]
    scores = {"iou": overall["iou"], "precision": overall["precision"],
              "recall": overall["recall"], "f1": overall["f1"]}
    scores.update({f"iou_{c}": m["iou"] for c, m in by_case.items() if c != "overall"})
    return scores


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pipe = build_pipeline(cfg)
    runs = list(cfg.eval.runs)
    label = (",".join(d.stat for d in pipe.detectors) + f" | {cfg.combine.name}")
    synthetic = bool(cfg.eval.get("synthetic", True))
    kind = "synthetic" if synthetic else "human-mask"
    print(f"=== {label}  (runs {runs}, {kind} eval) ===")

    if synthetic:
        scores = _synthetic_scores(cfg, pipe, runs, os.getcwd())
        mean = scores
        print(f"  OVERALL: IoU {mean['iou']:.3f}  prec {mean['precision']:.3f}  "
              f"rec {mean['recall']:.3f}  f1 {mean['f1']:.3f}  "
              f"(original mask IoU {scores.get('iou_original', float('nan')):.3f})")
        extra = {"mean_f1": mean["f1"]}
        extra.update({k: v for k, v in scores.items() if k.startswith("iou_")})
    else:
        metrics = evaluate(pipe, runs, verbose=True)
        mean = metrics["mean"]
        print(f"  MEAN: IoU {mean['iou']:.3f}  prec {mean['precision']:.3f}  "
              f"rec {mean['recall']:.3f}  (residual IoU {mean['residual_iou']:.3f})")
        extra = {"mean_residual_iou": mean["residual_iou"]}

    # Flatten the swept overrides for the results row.
    row = {"detectors": label, "combine": cfg.combine.name, "eval": kind,
           "mean_iou": mean["iou"], "mean_precision": mean["precision"],
           "mean_recall": mean["recall"], **extra}
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