#!/usr/bin/env python3
"""
masking.py -- automated masking pipeline for the xppl1016922 Jungfrau1M detector.

Runs in assembled space on the arrays a `Sample` carries (see automask/sample.py).

The pipeline is three separable, independently-registered stages, each living in
its own folder (one file per method):

    STATISTICS    (stats/)          Sample -> continuous z-field, or a mask
                                    directly (kind="pick" shape detectors, and
                                    the kind="floor" geometry/status masks)
    REGULARIZATION(regularization/) field->field (TV) or mask->mask (pad)
    COMBINATION   (combine/)        fuse the evidence channels onto the floor

The three registries below are the catalogue of everything available. Each is a
dict name -> spec, populated by importing the component packages:

    STATS         variance, mad_variance, blackhat, sigma_clipping,
                  hough_lines (pick), geometry (floor), status_as_mask (floor)
    REGULARIZERS  tv (field), frangi (field), pad (mask)
    COMBINERS     union (picks), weighted_sum (fields), mahalanobis (fields)

A Pipeline is ONE list of Channels plus a combiner. Which channels form the
intensity-free floor is not a second list to keep in step -- it is read from each
stat's registered `kind`, so a floor channel is configured exactly like any
other and its knobs are reachable the same way.
"""
from __future__ import annotations
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
from automask.sample import Sample  # noqa: F401


# ==========================================================================
#  Channel -- one stat + field-reg + threshold + mask-reg -> boolean pick
# ==========================================================================
@dataclass
class Channel:
    """A single evidence channel: one statistic and the stages around it.

    For a kind="field" stat the chain is the full one -- statistic ->
    field-regularizer -> threshold -> mask-regularizer -- and `params` supplies
    both the stat inputs and the threshold (k, mode).

    A kind="pick" stat (`hough_lines`) and a kind="floor" stat (`geometry`,
    `status_as_mask`) already return the boolean decision, so only the
    mask-regularizer applies; `field_reg` must be None and `field()` is an error.
    Nothing about them is silently ignored: setting a field-regularizer raises
    rather than being dropped, because the failure it would cause otherwise is
    invisible (TV on a 0/1 indicator flattens it below any threshold and the
    channel quietly returns nothing).

    Both regularizer slots accept either a single name or a LIST of names, applied
    left to right, with `*_params` given in the same shape (a bare params object
    for a single name, a list aligned with the names otherwise; `None` anywhere
    means that stage's defaults). Composition is what several methods actually
    need -- `asic_polish` wants `blob_scale` to aggregate before thresholding, and
    a thresholded graded field wants `fill_holes` then `area_gate` after -- and
    keeping each step a registered, separately-parameterised stage is what keeps
    it visible to the sweep driver instead of hard-coded inside a stat.

    `name` labels the channel when one stat appears more than once in a pipeline
    (two black-hats at different radii); it defaults to the stat name.
    """
    stat: str
    params: object = None
    field_reg: Optional[str | List[str]] = "tv"
    field_reg_params: object = None
    mask_reg: Optional[str | List[str]] = None
    mask_reg_params: object = None
    name: Optional[str] = None

    def __post_init__(self):
        spec = STATS[self.stat]
        if spec.emits_mask and self._stages(self.field_reg, self.field_reg_params):
            raise ValueError(
                f"stat '{self.stat}' is kind='{spec.kind}': it emits a boolean "
                f"mask, so there is no field for field_reg={self.field_reg!r} to "
                f"act on. Pass field_reg=None (Hydra: regularization=none).")
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

    @property
    def label(self) -> str:
        return self.name or self.stat

    @property
    def is_floor(self) -> bool:
        """Floor channels are the intensity-free, 100%-precision ones."""
        return STATS[self.stat].kind == "floor"

    def _params(self):
        return self.params if self.params is not None else STATS[self.stat].params()

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
        """Boolean pick, mask-regularized. A pick/floor stat supplies the mask
        directly; a field stat is thresholded at (k, mode) to get one.

        Evidence channels are gated by `real` -- they measure intensity, and a
        pixel with no value carries no evidence. Floor channels are NOT: the
        detector's dead pixels and the unmapped canvas are exactly what they are
        there to mask, so intersecting them with `real` would erase them.
        """
        spec, p = STATS[self.stat], self._params()
        if spec.emits_mask:
            m = np.asarray(spec.compute(sample, p), dtype=bool)
        else:
            mode = getattr(p, "mode", spec.mode)
            m = threshold_stat(self.field(sample), p.k, mode)
        gate = (lambda mask: mask) if self.is_floor else (lambda mask: mask & sample.real)
        m = gate(m)
        for name, params in self._stages(self.mask_reg, self.mask_reg_params):
            m = gate(REGULARIZERS[name].apply(m, params))
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
    """One list of channels plus a combiner.

    The floor is not a separate list: `floor_channels` reads it off each stat's
    registered `kind`, so adding `Channel("status_as_mask", StatusAsMaskParams(pad=3))`
    both puts it in the floor and makes its knobs reachable -- the old parallel
    list of bare names could do neither.
    """
    channels: List[Channel] = field(default_factory=list)
    combiner: str = "union"
    combiner_params: object = None

    def __post_init__(self):
        labels = [c.label for c in self.channels]
        duplicates = sorted({l for l in labels if labels.count(l) > 1})
        if duplicates:
            raise ValueError(
                f"channel labels must be unique, got duplicates {duplicates}; "
                f"pass Channel(..., name=...) to tell them apart.")

    @property
    def floor_channels(self) -> List[Channel]:
        return [c for c in self.channels if c.is_floor]

    @property
    def evidence_channels(self) -> List[Channel]:
        return [c for c in self.channels if not c.is_floor]

    def needs(self) -> Tuple[str, ...]:
        """Every Sample array this pipeline's statistics read, by name.

        These are resolved by `Sample.from_store`: a name that is a reduction is
        computed over the selected shots, anything else is a psana calibration
        accessor. Nothing here enumerates which is which.
        """
        wanted = set()
        for channel in self.channels:
            wanted |= set(STATS[channel.stat].needs)
        return tuple(sorted(wanted))

    def floor(self, sample) -> np.ndarray:
        """OR of the intensity-free floor channels; 100%-precision."""
        floor = np.zeros(sample.real.shape, dtype=bool)
        for channel in self.floor_channels:
            floor = floor | channel.pick(sample)
        return floor

    def run(self, sample, floor=None) -> np.ndarray:
        """Final boolean mask (True == masked).

        ``floor`` may supply a fixed boolean floor for perturbation studies.
        Normal masking leaves it unset and computes the floor from ``sample``.
        """
        floor = self.floor(sample) if floor is None else np.asarray(floor)
        if floor.dtype != np.bool_:
            raise TypeError("a pipeline floor must be boolean")
        if floor.shape != sample.real.shape:
            raise ValueError(
                f"pipeline floor shape {floor.shape} != sample shape {sample.real.shape}")
        cspec = COMBINERS[self.combiner]
        evidence = self.evidence_channels
        if cspec.consumes == "picks":
            comps = {c.label: c.pick(sample) for c in evidence}
        else:
            comps = {c.label: c.defectiveness(sample) for c in evidence}
        return cspec.combine(floor, comps, sample, self.combiner_params)


# ==========================================================================
#  production pipeline
# ==========================================================================
# z-worth of one hough_lines pixel for the consumes="fields" combiners. Chosen
# to sit above weighted_sum's k=3.5 so the channel masks on its own there, as it
# does under union -- the two recipes then differ in HOW evidence is fused, not in
# which channels can act. A pick has no measured z-scale, so this is a modelling
# choice; it is here, in the recipe, rather than defaulted in the stat.
_HOUGH_FUSION_Z = 5.0


def floor_channels() -> List[Channel]:
    """The intensity-free floor: unmapped/ASIC geometry + psana pixel status."""
    return [Channel("geometry", field_reg=None), Channel("status_as_mask", field_reg=None)]


def production_pipeline(combiner: str = "union",
                        line_detector: bool = True) -> Pipeline:
    """The default recipe: TV variance + hough_lines + asic_polish on the
    geometry + pixel-status floor. combiner="union" reproduces `combo`;
    combiner="weighted_sum" reproduces `combo_sum`.
    """
    from automask.stats.variance import VarianceParams
    from automask.stats.hough_lines import HoughLinesParams
    from automask.stats.asic_polish import AsicPolishParams
    from automask.regularization.tv import TVParams
    from automask.regularization.blob_scale import BlobScaleParams
    from automask.regularization.fill_holes import FillHolesParams
    from automask.regularization.area_gate import AreaGateParams
    from automask.combine.weighted_sum import WeightedSumParams

    channels = floor_channels()
    channels.append(Channel("variance", VarianceParams(k=3.5, mode="low"),
                            field_reg="tv", field_reg_params=TVParams(4.0)))
    if line_detector:
        channels.append(Channel(
            "hough_lines", HoughLinesParams(defectiveness_scale=_HOUGH_FUSION_Z),
            field_reg=None))
    channels.append(Channel(
        "asic_polish", AsicPolishParams(asic=256, n_iter=3, k=15.0, mode="high"),
        field_reg=["blob_scale"], field_reg_params=[BlobScaleParams()],
        mask_reg=["fill_holes", "area_gate"],
        mask_reg_params=[FillHolesParams(), AreaGateParams()]))
    if combiner == "weighted_sum":
        return Pipeline(channels, combiner="weighted_sum",
                        combiner_params=WeightedSumParams(k=3.5, pad=2))
    return Pipeline(channels, combiner=combiner)


# ==========================================================================
#  single-image adapter (`module:function` masker convention)
# ==========================================================================
def mask_image(
    image: np.ndarray,
    *,
    blackhat_radius: int = 5,
    blackhat_k: float = 6.0,
    blackhat_weight: float = 1.0,
    pad: int = 2,
) -> np.ndarray:
    """Honest single-image subset of the run-level pipeline: the geometry floor
    plus a TV+pad black-hat channel unioned onto it. True == masked.

    This runs the same Pipeline every other entry point runs, on a Sample holding
    one image -- not a second, hand-inlined copy of the recipe. Two channels of
    the production recipe are necessarily absent: `variance` needs many frames,
    and `status_as_mask` is a per-run psana constant a bare image cannot identify.
    Use `production_pipeline()` on a `Sample.from_store` run for the full floor.
    """
    from automask.stats.blackhat import BlackhatParams
    from automask.regularization.tv import TVParams
    from automask.regularization.pad import PadParams

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"mask_image expects a 2-D array, got {image.shape}")
    work = np.where(np.isfinite(image), image, 0.0).astype(np.float64)
    if not work.any():
        return np.ones(image.shape, dtype=bool)

    return Pipeline([
        Channel("geometry", field_reg=None),
        Channel("blackhat",
                BlackhatParams(radius=blackhat_radius, k=blackhat_k, mode="high"),
                field_reg="tv", field_reg_params=TVParams(blackhat_weight),
                mask_reg="pad", mask_reg_params=PadParams(pad)),
    ]).run(Sample(run=-1, arrays={"mean": work}))
