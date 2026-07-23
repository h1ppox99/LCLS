"""
exp_selection.py -- Direction 2: shot-selection / normalization effect on masking.

Recomputes the lit features (umean/ustd) under selection variants and measures
the production pipeline's real-mask IoU on both runs (the production objective,
and the quantity most sensitive to feature quality). Feature computes hit the
FeatureStore cache when warm, else one XTC pass per (run, selection).

    source psana_env.sh
    python -m automask.studies.exploration.exp_selection
"""
from __future__ import annotations

from automask.masking import production_pipeline
from automask.shot_selection import ShotSelection
from automask.studies.exploration.harness import sample_custom, score_sample

RUNS = (389, 475)

VARIANTS = {
    "on_all_none":  ShotSelection(xray="on"),                          # = catalogue (warm)
    "on_all_ipm2":  ShotSelection(xray="on", normalization="ipm2"),
    "on_n500_none": ShotSelection(xray="on", n_shots=500),
    "on_n200_none": ShotSelection(xray="on", n_shots=200),
    "on_notrim":    ShotSelection(xray="on", filter_low=0.0, filter_high=0.0),
}


def main():
    pipe = production_pipeline("union")
    print(f"=== Direction 2: selection/normalization -> real IoU | runs {RUNS} ===")
    print(f"{'variant':14s} | " + " ".join(f"IoU{r:>4}  prec{r:>3}  rec{r:>4}" for r in RUNS)
          + " |   mean IoU")
    for name, sel in VARIANTS.items():
        line, ious = "", []
        for run in RUNS:
            s = sample_custom(run, sel)
            sc = score_sample(pipe, s)
            ious.append(sc["iou"])
            line += f"  {sc['iou']:.3f}  {sc['precision']:.3f}  {sc['recall']:.3f}"
        mean_iou = sum(ious) / len(ious)
        print(f"{name:14s} | {line} |   {mean_iou:.3f}")
    print("SELECTION-EXP-DONE")


if __name__ == "__main__":
    main()
