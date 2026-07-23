#!/usr/bin/env python3
"""
masking.py -- automated masking pipeline for the xppl1016922 Jungfrau1M detector.

Runs in assembled space (1064, 1030) on the frozen numpy dataset in ./data.

The pipeline is three separable, independently-registered stages, each living in
its own folder (one file per method), plus an intensity-free floor:

    STATISTICS    (stats/)          raw Sample -> continuous z-field (or floor mask)
    REGULARIZATION(regularization/) field->field (TV) or mask->mask (pad/close_open)
    COMBINATION   (combine/)        fuse per-detector evidence onto the floor

The three registries below are the catalogue of everything available. Each is a
dict name -> spec, populated by importing the component packages:

    STATS         variance, window_median, blackhat, radial_median,
                  azimuthal_sigma, geometry (floor), calib (floor)
    REGULARIZERS  tv (field), frangi (field), pad (mask), close_open (mask)
    COMBINERS     union (picks), weighted_sum (fields), mahalanobis (fields)

A Pipeline composes floor stats + a list of Detectors + one combiner. Sweeping is
done by studies/sweep_hyperparameters.py (Hydra); this module is the library +
the production `main()`.

Run:  python -m automask.masking
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Importing the component packages registers every method into the three dicts.
from automask import stats  # noqa: F401
from automask import regularization  # noqa: F401
from automask import combine  # noqa: F401
from automask.stats.base import STATS, threshold_stat, robust_z          # noqa: F401
from automask.regularization.base import REGULARIZERS
from automask.combine.base import COMBINERS
from automask.evaluation import Sample, load_sample, evaluate, EVAL_RUNS  # noqa: F401
from automask.dataset import score

HERE = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.join(HERE, "outputs", "masks")
os.makedirs(MASK_DIR, exist_ok=True)


# ==========================================================================
#  Detector -- one stat + field-reg + threshold + mask-reg -> boolean pick
# ==========================================================================
@dataclass
class Detector:
    """A single evidence channel: statistic -> field-regularizer -> threshold ->
    mask-regularizer. `stat_params` (a stat's Params dataclass) supplies both the
    stat inputs and the threshold (k, mode); None uses the stat's defaults."""
    stat: str
    stat_params: object = None
    field_reg: Optional[str] = "tv"
    field_reg_params: object = None
    mask_reg: Optional[str] = None
    mask_reg_params: object = None

    def _params(self):
        return self.stat_params if self.stat_params is not None else STATS[self.stat].params()

    def field(self, sample) -> np.ndarray:
        """Continuous statistic field, field-regularized (e.g. TV-denoised)."""
        z = STATS[self.stat].compute(sample, self._params())
        if self.field_reg:
            z = REGULARIZERS[self.field_reg].apply(z, self.field_reg_params)
        return z

    def pick(self, sample) -> np.ndarray:
        """Boolean pick: threshold the (regularized) field, gate by real, mask-reg."""
        spec, p = STATS[self.stat], self._params()
        mode = getattr(p, "mode", spec.mode)
        m = threshold_stat(self.field(sample), p.k, mode) & sample.real
        if self.mask_reg:
            m = REGULARIZERS[self.mask_reg].apply(m, self.mask_reg_params) & sample.real
        return m

    def defectiveness(self, sample) -> np.ndarray:
        """Sign-aligned DEFECTIVENESS field (large > 0 == wants masking), 0 outside
        `real`. Folds low/both stats so every field points the same way for the
        continuous-fusion combiners (weighted_sum, mahalanobis)."""
        spec, p = STATS[self.stat], self._params()
        mode = getattr(p, "mode", spec.mode)
        z = self.field(sample)
        d = -z if mode == "low" else (np.abs(z) if mode == "both" else z)
        d = np.array(d, dtype=np.float64, copy=True)
        d[~sample.real] = 0.0
        return d


# ==========================================================================
#  Pipeline -- floor stats + detectors + combiner -> final mask
# ==========================================================================
@dataclass
class Pipeline:
    detectors: List[Detector] = field(default_factory=list)
    floor_stats: List[str] = field(default_factory=lambda: ["geometry", "calib"])
    combiner: str = "union"
    combiner_params: object = None

    def features_needed(self) -> Tuple[str, ...]:
        """Shot-selection features this pipeline's stats require, from their
        declared `needs` (floor stats + detectors), restricted to the feature
        catalogue. This is what the evaluation loop materializes -- the masking
        strategy decides which features get built from XTC."""
        from automask.features import FEATURES
        wanted = set()
        for name in self.floor_stats:
            wanted |= set(STATS[name].needs)
        for d in self.detectors:
            wanted |= set(STATS[d.stat].needs)
        return tuple(sorted(n for n in wanted if n in FEATURES))

    def floor(self, sample) -> np.ndarray:
        """OR of the intensity-free floor stats (geometry + calib), 100%-precision."""
        f = np.zeros(sample.sumimg.shape, dtype=bool)
        for name in self.floor_stats:
            f = f | STATS[name].compute(sample, None)
        return f

    def run(self, sample) -> np.ndarray:
        """Final boolean mask (True == masked)."""
        floor = self.floor(sample)
        cspec = COMBINERS[self.combiner]
        if cspec.consumes == "picks":
            comps = {d.stat: d.pick(sample) for d in self.detectors}
        else:
            comps = {d.stat: d.defectiveness(sample) for d in self.detectors}
        return cspec.combine(floor, comps, sample, self.combiner_params)


# ==========================================================================
#  production pipeline -- the live 3-detector recipe (regression anchor)
# ==========================================================================
def production_pipeline(combiner: str = "union") -> Pipeline:
    """The current production recipe: TV variance + TV+pad window-median + TV+pad
    black-hat on the geometry+calib floor. combiner="union" reproduces `combo`;
    combiner="weighted_sum" reproduces `combo_sum`."""
    from automask.stats.variance import VarianceParams
    from automask.stats.window_median import WindowMedianParams
    from automask.stats.blackhat import BlackhatParams
    from automask.regularization.tv import TVParams
    from automask.regularization.pad import PadParams
    from automask.combine.weighted_sum import WeightedSumParams

    detectors = [
        Detector("variance", VarianceParams(k=3.5, mode="low"),
                 field_reg="tv", field_reg_params=TVParams(4.0), mask_reg=None),
        Detector("window_median", WindowMedianParams(win=21, k=5.0, mode="low"),
                 field_reg="tv", field_reg_params=TVParams(1.0),
                 mask_reg="pad", mask_reg_params=PadParams(2)),
        Detector("blackhat", BlackhatParams(radius=5, k=6.0, mode="high"),
                 field_reg="tv", field_reg_params=TVParams(1.0),
                 mask_reg="pad", mask_reg_params=PadParams(2)),
    ]
    if combiner == "weighted_sum":
        return Pipeline(detectors, combiner="weighted_sum",
                        combiner_params=WeightedSumParams(k=3.5, pad=2))
    return Pipeline(detectors, combiner=combiner)


def best_pipeline(combiner: str = "union") -> Pipeline:
    """`production_pipeline` plus the `dead_holes` + `stuck` floors validated in
    exploration.MD D1: same detectors and same real-mask IoU/precision as
    production, but structurally catches isolated dead / stuck / dead-column
    pixels that the frozen `calib` mask can miss on a new run. Zero regression on
    runs 389/475; strictly more robust run-agnostically."""
    p = production_pipeline(combiner)
    return Pipeline(p.detectors, floor_stats=["geometry", "calib", "dead_holes", "stuck"],
                    combiner=p.combiner, combiner_params=p.combiner_params)


# ==========================================================================
#  single-image adapter (used by review.py --masker masking:mask_image)
# ==========================================================================
def mask_image(
    image: np.ndarray,
    *,
    win: int = 21,
    window_k: float = 5.0,
    window_weight: float = 1.0,
    blackhat_radius: int = 5,
    blackhat_k: float = 6.0,
    blackhat_weight: float = 1.0,
    pad: int = 2,
) -> np.ndarray:
    """Honest single-image subset of the run-level pipeline: invalid pixels and
    geometry lines form the floor, then TV+pad window-median and black-hat picks
    are unioned onto it. The variance detector is absent (per-pixel variance needs
    many frames) and so is the run-specific calib mask. True == masked."""
    from automask.stats.geometry import geometry_mask
    from automask.stats.window_median import window_median_stat
    from automask.stats.blackhat import blackhat_stat
    from automask.regularization.tv import tv_denoise
    from automask.regularization.pad import pad_mask
    from automask.combine.union import combine_masks

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"mask_image expects a 2-D array, got {image.shape}")
    work = image.astype(np.float64, copy=False)
    finite = np.isfinite(work)
    if not finite.any():
        return np.ones(image.shape, dtype=bool)

    carrying_data = finite & (work != 0)
    floor = ~finite | geometry_mask(carrying_data, pad=pad)
    wm = tv_denoise(window_median_stat(work, finite, win=win), window_weight)
    wm = pad_mask(threshold_stat(wm, window_k, "low") & finite, pad) & finite
    bh = tv_denoise(blackhat_stat(work, finite, radius=blackhat_radius), blackhat_weight)
    bh = pad_mask(threshold_stat(bh, blackhat_k, "high") & finite, pad) & finite
    return combine_masks(floor, {"window_median": wm, "blackhat": bh}).astype(bool, copy=False)


# ==========================================================================
#  main -- production report (regression anchor, run 475)
# ==========================================================================
def main():
    RUN = 475
    pipe = production_pipeline("union")
    pipe_sum = production_pipeline("weighted_sum")
    sample = load_sample(RUN, features=pipe.features_needed())

    floor = pipe.floor(sample)
    human = sample.human
    target = human & ~floor
    sf = score(floor, human)
    print(f"=== geometry + calib floor (run {RUN}) ===")
    print(f"  floor vs human : {100*floor.mean():.2f}% masked  IoU {sf['iou']:.3f}  "
          f"prec {sf['precision']:.3f}  rec {sf['recall']:.3f}")
    print(f"  residual target = human & ~floor : {int(target.sum())} px to find\n")

    picks = {d.stat: d.pick(sample) for d in pipe.detectors}
    combo = pipe.run(sample)
    combo_sum = pipe_sum.run(sample)

    print(f"{'detector':16s} | {'added%':>6s} {'T-prec':>6s} {'T-rec':>6s} "
          f"| {'IoU':>7s} {'prec':>6s} {'rec':>6s}")
    print("-" * 70)
    for name, M in {**picks, "combo": combo, "combo_sum": combo_sum}.items():
        Mo = M & ~floor
        st, sc = score(Mo, target), score(floor | M, human)
        print(f"{name:16s} | {100*Mo.mean():5.2f}% {st['precision']:6.3f} "
              f"{st['recall']:6.3f} | {sc['iou']:7.3f} {sc['precision']:6.3f} "
              f"{sc['recall']:6.3f}")
    print(f"\n(reference: floor alone -> IoU {sf['iou']:.3f})")

    np.save(os.path.join(MASK_DIR, f"geometry_mask_run{RUN:04d}.npy"), floor)
    np.save(os.path.join(MASK_DIR, f"combo_run{RUN:04d}.npy"), combo)
    np.save(os.path.join(MASK_DIR, f"combo_sum_run{RUN:04d}.npy"), combo_sum)


if __name__ == "__main__":
    main()
