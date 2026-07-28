#!/usr/bin/env python3
"""
masking.py -- automated masking pipeline for the xppl1016922 Jungfrau1M detector.

Runs in assembled space (1064, 1030) on the frozen numpy dataset in ./data.

The pipeline is three separable, independently-registered stages, each living in
its own folder (one file per method), plus an intensity-free floor:

    STATISTICS    (stats/)          raw Sample -> continuous z-field, or a mask
                                    directly (kind="pick" shape detectors, and
                                    the kind="floor" geometry masks)
    REGULARIZATION(regularization/) field->field (TV) or mask->mask (pad/close_open)
    COMBINATION   (combine/)        fuse per-detector evidence onto the floor

The three registries below are the catalogue of everything available. Each is a
dict name -> spec, populated by importing the component packages:

    STATS         variance, mad_variance, blackhat, sigma_clipping,
                  hough_lines (pick), geometry (floor), calib (floor)
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
    """A single evidence channel.

    For a kind="field" stat the chain is the full one -- statistic ->
    field-regularizer -> threshold -> mask-regularizer -- and `stat_params`
    supplies both the stat inputs and the threshold (k, mode).

    For a kind="pick" stat (`hough_lines`) the statistic already returns the
    boolean decision, so only the mask-regularizer applies; `field_reg` must be
    None and `field()` is an error. Nothing about a pick is silently ignored:
    setting a field-regularizer on one raises rather than being dropped, because
    the failure it would cause otherwise is invisible (TV on a 0/1 indicator
    flattens it below any threshold and the detector quietly returns nothing).

    Both regularizer slots accept either a single name or a LIST of names, applied
    left to right, with `*_params` given in the same shape (a bare params object
    for a single name, a list aligned with the names otherwise; `None` anywhere
    means that stage's defaults). Composition is what several methods actually
    need -- `asic_polish` wants `blob_scale` to aggregate before thresholding, and
    a thresholded graded field wants `fill_holes` then `area_gate` after -- and
    keeping each step a registered, separately-parameterised stage is what keeps
    it visible to the sweep driver instead of hard-coded inside a stat.
    """
    stat: str
    stat_params: object = None
    field_reg: Optional[str | List[str]] = "tv"
    field_reg_params: object = None
    mask_reg: Optional[str | List[str]] = None
    mask_reg_params: object = None

    def __post_init__(self):
        spec = STATS[self.stat]
        if spec.kind == "pick" and self._stages(self.field_reg, self.field_reg_params):
            raise ValueError(
                f"stat '{self.stat}' is kind='pick': it emits a boolean mask, so "
                f"there is no field for field_reg={self.field_reg!r} to act on. "
                f"Pass field_reg=None (Hydra: regularization=none).")
        if spec.kind == "floor":
            raise ValueError(
                f"stat '{self.stat}' is a floor stat -- put it in "
                f"Pipeline.floor_stats, not in a Detector.")
        for names, params, kind in ((self.field_reg, self.field_reg_params, "field"),
                                    (self.mask_reg, self.mask_reg_params, "mask")):
            for name, _ in self._stages(names, params):
                got = REGULARIZERS[name].kind
                if got != kind:
                    raise ValueError(
                        f"regularizer '{name}' is kind='{got}' but was given as a "
                        f"{kind}_reg on stat '{self.stat}'; a {got} regularizer "
                        f"acts on {'a continuous field' if got == 'field' else 'a boolean mask'}.")

    @staticmethod
    def _stages(names, params) -> List[tuple]:
        """Normalize a regularizer slot to an ordered [(name, params), ...] list.

        Accepts None, a single name, or a list of names; `params` may be None, a
        single params object, or a list aligned with `names`."""
        if not names:
            return []
        if isinstance(names, str):
            return [(names, params)]
        names = list(names)
        if params is None:
            params = [None] * len(names)
        elif not isinstance(params, (list, tuple)):
            raise ValueError(
                f"regularizer list {names} needs a list of params (or None), "
                f"got a single {type(params).__name__}")
        if len(params) != len(names):
            raise ValueError(
                f"regularizer list {names} has {len(names)} entries but "
                f"{len(params)} params")
        return list(zip(names, params))

    def _params(self):
        return self.stat_params if self.stat_params is not None else STATS[self.stat].params()

    def field(self, sample) -> np.ndarray:
        """Continuous statistic field, field-regularized (e.g. TV-denoised)."""
        spec = STATS[self.stat]
        if spec.emits_mask:
            raise TypeError(
                f"stat '{self.stat}' is kind='{spec.kind}' and has no continuous "
                f"field; use .pick(sample).")
        z = spec.compute(sample, self._params())
        for name, params in self._stages(self.field_reg, self.field_reg_params):
            z = REGULARIZERS[name].apply(z, params)
        return z

    def pick(self, sample) -> np.ndarray:
        """Boolean pick, gated by `real` and mask-regularized. A pick stat supplies
        the mask directly; a field stat is thresholded at (k, mode) to get one."""
        spec, p = STATS[self.stat], self._params()
        if spec.kind == "pick":
            m = np.asarray(spec.compute(sample, p), dtype=bool) & sample.real
        else:
            mode = getattr(p, "mode", spec.mode)
            m = threshold_stat(self.field(sample), p.k, mode) & sample.real
        for name, params in self._stages(self.mask_reg, self.mask_reg_params):
            m = REGULARIZERS[name].apply(m, params) & sample.real
        return m

    def defectiveness(self, sample) -> np.ndarray:
        """Sign-aligned DEFECTIVENESS field (large > 0 == wants masking), 0 outside
        `real`. Folds low/both stats so every field points the same way for the
        continuous-fusion combiners (weighted_sum, mahalanobis).

        A pick stat has no z-scale of its own -- its mask is a decision, not a
        measurement -- so fusing it means choosing what one masked pixel is worth
        on the other detectors' robust-z scale. That choice is required to be
        explicit (`defectiveness_scale` on the stat's params), because the default
        of 1.0 an indicator would imply is silently below every sensible fusion
        threshold: `weighted_sum` cuts at k=3.5, so an unscaled pick could never
        carry a pixel and the detector would vanish from the mask with no error."""
        spec, p = STATS[self.stat], self._params()
        if spec.kind == "pick":
            scale = getattr(p, "defectiveness_scale", None)
            if scale is None:
                raise ValueError(
                    f"stat '{self.stat}' is kind='pick' and carries no robust-z "
                    f"scale, so it cannot be fused by a consumes='fields' "
                    f"combiner. Use combiner='union', or set "
                    f"{type(p).__name__}.defectiveness_scale to the z-value one "
                    f"picked pixel should be worth (must exceed the combiner's k "
                    f"to mask on its own).")
            d = np.where(self.pick(sample), float(scale), 0.0)
        else:
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
#  production pipeline 
# ==========================================================================
# z-worth of one hough_lines pixel for the consumes="fields" combiners. Chosen
# to sit above weighted_sum's k=3.5 so the detector masks on its own there, as it
# does under union -- the two recipes then differ in HOW evidence is fused, not in
# which detectors can act. A pick has no measured z-scale, so this is a modelling
# choice; it is here, in the recipe, rather than defaulted in the stat.
_HOUGH_FUSION_Z = 5.0


def production_pipeline(combiner: str = "union",
                        line_detector: bool = True) -> Pipeline:
    """The default recipe: TV variance + hough_lines + asic_polish on the
    geometry+calib floor. combiner="union" reproduces `combo`;
    combiner="weighted_sum" reproduces `combo_sum`.
    """
    from automask.stats.variance import VarianceParams
    from automask.stats.sigma_clipping import SigmaClippingParams
    from automask.stats.hough_lines import HoughLinesParams
    from automask.stats.asic_polish import AsicPolishParams
    from automask.regularization.tv import TVParams
    from automask.regularization.blob_scale import BlobScaleParams
    from automask.regularization.fill_holes import FillHolesParams
    from automask.regularization.area_gate import AreaGateParams
    from automask.combine.weighted_sum import WeightedSumParams

    detectors = [
        Detector("variance", VarianceParams(k=3.5, mode="low"),
                 field_reg="tv", field_reg_params=TVParams(4.0), mask_reg=None),
        # Detector("sigma_clipping", SigmaClippingParams(k=5.0, mode="both"),
        #          field_reg="tv", field_reg_params=TVParams(1.0), mask_reg=None),
    ]
    if line_detector:
        detectors.append(Detector(
            "hough_lines", HoughLinesParams(defectiveness_scale=_HOUGH_FUSION_Z),
            field_reg=None, mask_reg=None))
    detectors.append(Detector(
        "asic_polish", AsicPolishParams(asic=256, n_iter=3, k=15.0, mode="high"),
        field_reg=["blob_scale"], field_reg_params=[BlobScaleParams()],
        mask_reg=["fill_holes", "area_gate"],
        mask_reg_params=[FillHolesParams(), AreaGateParams()]))
    if combiner == "weighted_sum":
        return Pipeline(detectors, combiner="weighted_sum",
                        combiner_params=WeightedSumParams(k=3.5, pad=2))
    return Pipeline(detectors, combiner=combiner)


# ==========================================================================
#  single-image adapter (used by review.py --masker masking:mask_image)
# ==========================================================================
def mask_image(
    image: np.ndarray,
    *,
    calib: Optional[np.ndarray] = None,
    blackhat_radius: int = 5,
    blackhat_k: float = 6.0,
    blackhat_weight: float = 1.0,
    pad: int = 2,
) -> np.ndarray:
    """Honest single-image subset of the run-level pipeline: invalid pixels and
    geometry lines form the floor, then a TV+pad black-hat pick is unioned onto
    it. The variance detector is absent (per-pixel variance needs many frames).

    `calib` is the run's psana pixel-status bad-pixel mask (a boolean array in
    the same assembled space as `image`, True == masked) -- the single image is
    built from that run, so its dead pixels apply. Pass e.g.
    `load_mask("statusMask_run0475_asm")`; when omitted the floor is geometry
    only. True == masked."""
    from automask.stats.geometry import geometry_mask
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
    if calib is not None:
        calib = np.asarray(calib, dtype=bool)
        if calib.shape != image.shape:
            raise ValueError(
                f"calib mask shape {calib.shape} != image shape {image.shape}")
        floor = floor | calib
    bh = tv_denoise(blackhat_stat(work, finite, radius=blackhat_radius), blackhat_weight)
    bh = pad_mask(threshold_stat(bh, blackhat_k, "high") & finite, pad) & finite
    return combine_masks(floor, {"blackhat": bh}).astype(bool, copy=False)


# ==========================================================================
#  main -- default recipe report, every evaluation run
# ==========================================================================
def _check_experiment_config():
    """Warn if conf/experiment/production.yaml has drifted from the Python recipe.

    The Hydra experiment duplicates `production_pipeline` as data, and the README
    points users at it -- so a silent divergence means the documented production
    command runs a different mask than the library does. That had already
    happened once (the yaml still named sigma_clipping long after main() moved to
    mad_variance), which is exactly the failure this catches. Compares the stat
    names only: knob-level drift is the sweep's whole point."""
    path = os.path.join(HERE, "conf", "experiment", "production.yaml")
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg_stats = [d["stat"] for d in cfg.get("detectors", [])]
    except Exception as e:                      # config is optional at runtime
        print(f"  [warn] could not read {path}: {e}")
        return
    py_stats = [d.stat for d in production_pipeline("union").detectors]
    if cfg_stats != py_stats:
        print(f"  [warn] conf/experiment/production.yaml is out of step with "
              f"production_pipeline(): yaml={cfg_stats} python={py_stats}")


def report(RUN: int):
    """Per-detector + combined scores for one run, with agreement figures."""
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

    # One agreement figure per detector plus the combined masks, so each
    # channel's contribution is visible rather than only tabulated.
    from automask import viz
    fig_dir = os.path.join(HERE, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for name, M in {**picks, "combo": combo, "combo_sum": combo_sum}.items():
        out = os.path.join(fig_dir, f"{name}_run{RUN:04d}.png")
        viz.save_agreement(floor | M, floor, human, RUN, out,
                           title=f"run {RUN} — {name}")
        print(f"[figure] {out}")

    panels = os.path.join(fig_dir, f"panels_run{RUN:04d}.png")
    viz.detector_panels(pipe, sample, out=panels)
    print(f"[figure] {panels}")


def main(runs=None):
    """Report the production recipe on every evaluation run."""
    _check_experiment_config()
    for run in (EVAL_RUNS if runs is None else runs):
        report(run)
        print()


if __name__ == "__main__":
    main()
