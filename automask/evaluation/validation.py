"""Run-local validation under explicit data and model perturbations."""

from __future__ import annotations

import itertools
import numbers
from dataclasses import dataclass, fields, replace
from typing import Any, Optional, Tuple

import numpy as np

from automask.combine.base import COMBINERS
from automask.evaluation.metrics import compare_masks, instability
from automask.evaluation.report import (
    CaseDelta,
    ChannelResult,
    EnsembleResult,
    MaskValidationReport,
)
from automask.image_store import REDUCTIONS, ImageStore
from automask.masking import Channel, Pipeline
from automask.regularization.base import REGULARIZERS
from automask.sample import DERIVED, Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection
from automask.stats.base import STATS


@dataclass(frozen=True)
class ParameterSweep:
    """Explicit nearby values for one stable pipeline parameter path."""

    path: str
    values: Tuple[Any, ...]
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))
        if not self.path or not self.values:
            raise ValueError("a parameter sweep needs a path and at least one value")
        if not self.reason or "\n" in self.reason:
            raise ValueError("a parameter sweep reason must be one non-empty line")

    def as_dict(self) -> dict:
        return {"path": self.path, "values": list(self.values), "reason": self.reason}


@dataclass(frozen=True)
class MaskValidationDesign:
    sweeps: Tuple[ParameterSweep, ...] = ()
    n_folds: int = 10
    strategies: Tuple[str, ...] = ("round_robin", "chronological")
    prune_redundant: bool = True

    def __post_init__(self):
        object.__setattr__(self, "sweeps", tuple(self.sweeps))
        object.__setattr__(self, "strategies", tuple(self.strategies))
        if not all(isinstance(sweep, ParameterSweep) for sweep in self.sweeps):
            raise TypeError("sweeps must contain ParameterSweep objects")
        if (
            not isinstance(self.n_folds, int)
            or isinstance(self.n_folds, bool)
            or self.n_folds < 2
        ):
            raise ValueError("n_folds must be an integer >= 2")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError(
                "strategies must be a non-empty sequence without duplicates"
            )
        unknown = set(self.strategies) - {"round_robin", "chronological"}
        if unknown:
            raise ValueError(f"unknown fold strategies {sorted(unknown)}")

    def as_dict(self) -> dict:
        return {
            "sweeps": [sweep.as_dict() for sweep in self.sweeps],
            "n_folds": self.n_folds,
            "strategies": list(self.strategies),
            "prune_redundant": self.prune_redundant,
        }


@dataclass(frozen=True)
class _Variant:
    label: str
    pipeline: Pipeline


def _channel(pipeline: Pipeline, label: str) -> Channel:
    matches = [channel for channel in pipeline.channels if channel.label == label]
    if len(matches) != 1:
        raise ValueError(f"parameter path channel {label!r} is not unique")
    return matches[0]


def _compatible(current, value) -> bool:
    if isinstance(current, (bool, np.bool_)):
        return isinstance(value, (bool, np.bool_))
    if isinstance(current, numbers.Integral):
        return isinstance(value, numbers.Integral) and not isinstance(
            value, (bool, np.bool_)
        )
    if isinstance(current, numbers.Real):
        return isinstance(value, numbers.Real) and not isinstance(
            value, (bool, np.bool_)
        )
    if isinstance(current, str):
        return isinstance(value, str)
    if isinstance(current, tuple):
        return isinstance(value, (tuple, list))
    if isinstance(current, list):
        return isinstance(value, (tuple, list))
    return current is None or isinstance(value, type(current))


def _replace_field(params, params_type, name: str, value):
    params = params if params is not None else params_type()
    names = {field.name for field in fields(params)}
    if name not in names:
        raise ValueError(f"{type(params).__name__} has no parameter {name!r}")
    current = getattr(params, name)
    if not _compatible(current, value):
        raise TypeError(
            f"parameter {name!r} expects a value compatible with "
            f"{type(current).__name__}, got {type(value).__name__}"
        )
    return replace(params, **{name: value})


def _stage_params(channel: Channel, kind: str, stage_name: str):
    names = getattr(channel, kind)
    params = getattr(channel, f"{kind}_params")
    stages = channel._stages(names, params)
    matches = [index for index, (name, _) in enumerate(stages) if name == stage_name]
    if len(matches) != 1:
        raise ValueError(
            f"channel {channel.label!r} does not have one {kind} regularizer "
            f"named {stage_name!r}"
        )
    return names, stages, matches[0]


def _get_parameter(pipeline: Pipeline, path: str):
    parts = path.split(".")
    if parts[:2] == ["combiner", "params"] and len(parts) == 3:
        params = pipeline.combiner_params or COMBINERS[pipeline.combiner].params()
        if parts[2] not in {field.name for field in fields(params)}:
            raise ValueError(f"combiner params have no parameter {parts[2]!r}")
        return getattr(params, parts[2])
    if len(parts) < 3:
        raise ValueError(f"invalid parameter path {path!r}")
    channel = _channel(pipeline, parts[0])
    if channel.is_floor:
        raise ValueError("the validation floor is fixed and cannot be swept")
    if parts[1] == "params" and len(parts) == 3:
        params = channel._params()
        if parts[2] not in {field.name for field in fields(params)}:
            raise ValueError(f"{type(params).__name__} has no parameter {parts[2]!r}")
        return getattr(params, parts[2])
    if parts[1] in ("field_reg", "mask_reg") and len(parts) == 4:
        _, stages, index = _stage_params(channel, parts[1], parts[2])
        params = stages[index][1] or REGULARIZERS[parts[2]].params()
        if parts[3] not in {field.name for field in fields(params)}:
            raise ValueError(f"{type(params).__name__} has no parameter {parts[3]!r}")
        return getattr(params, parts[3])
    raise ValueError(f"invalid parameter path {path!r}")


def _set_parameter(pipeline: Pipeline, path: str, value) -> Pipeline:
    parts = path.split(".")
    if parts[:2] == ["combiner", "params"] and len(parts) == 3:
        params = _replace_field(
            pipeline.combiner_params,
            COMBINERS[pipeline.combiner].params,
            parts[2],
            value,
        )
        return replace(pipeline, combiner_params=params)

    channel = _channel(pipeline, parts[0] if parts else "")
    if channel.is_floor:
        raise ValueError("the validation floor is fixed and cannot be swept")
    if len(parts) == 3 and parts[1] == "params":
        updated = replace(
            channel,
            params=_replace_field(
                channel.params, STATS[channel.stat].params, parts[2], value
            ),
        )
    elif len(parts) == 4 and parts[1] in ("field_reg", "mask_reg"):
        names, stages, index = _stage_params(channel, parts[1], parts[2])
        stage_params = _replace_field(
            stages[index][1], REGULARIZERS[parts[2]].params, parts[3], value
        )
        updated_stages = [params for _, params in stages]
        updated_stages[index] = stage_params
        shaped = updated_stages[0] if isinstance(names, str) else updated_stages
        updated = replace(channel, **{f"{parts[1]}_params": shaped})
    else:
        raise ValueError(f"invalid parameter path {path!r}")
    channels = [
        updated if item.label == channel.label else item for item in pipeline.channels
    ]
    return replace(pipeline, channels=channels)


def _variants(pipeline: Pipeline, design: MaskValidationDesign) -> Tuple[_Variant, ...]:
    variants = [_Variant("baseline", pipeline)]
    seen = set()
    for sweep in design.sweeps:
        baseline = _get_parameter(pipeline, sweep.path)
        for value in sweep.values:
            key = (sweep.path, repr(value))
            if key in seen:
                raise ValueError(f"duplicate perturbation {sweep.path}={value!r}")
            seen.add(key)
            if value == baseline:
                raise ValueError(
                    f"perturbation {sweep.path}={value!r} equals the baseline"
                )
            variants.append(
                _Variant(
                    f"{sweep.path}={value!r}",
                    _set_parameter(pipeline, sweep.path, value),
                )
            )
    return tuple(variants)


def _without_channels(pipeline: Pipeline, labels) -> Pipeline:
    labels = set(labels)
    return replace(
        pipeline,
        channels=[
            channel for channel in pipeline.channels if channel.label not in labels
        ],
    )


def _fold_samples(pipeline, run, selection, store, design, full_sample):
    reductions = sorted(({"mean", *pipeline.needs()} - set(DERIVED)) & REDUCTIONS)
    samples = {"full": full_sample}
    groups = {}
    for strategy in design.strategies:
        arrays = {
            name: store.folds(
                run, selection, name, n_folds=design.n_folds, strategy=strategy
            )
            for name in reductions
        }
        keys = []
        for index in range(design.n_folds):
            key = f"{strategy}:{index}"
            samples[key] = full_sample.with_arrays(
                **{name: values[index] for name, values in arrays.items()}
            )
            keys.append(key)
        groups[strategy] = tuple(keys)
    return samples, groups


def _evidence_mask(pipeline, sample, floor, domain):
    mask = np.asarray(pipeline.run(sample, floor=floor))
    if mask.dtype != np.bool_ or mask.shape != floor.shape:
        raise TypeError("a pipeline must return a boolean mask with the floor shape")
    if (floor & ~mask).any():
        raise ValueError("a pipeline final mask must contain the fixed floor")
    return mask & domain


def _compute_masks(variants, samples, floor, domain, cache=None):
    cache = {} if cache is None else cache
    masks = {}
    for variant_index, variant in enumerate(variants):
        signature = repr(variant.pipeline)
        for sample_key, sample in samples.items():
            cache_key = signature, sample_key
            if cache_key not in cache:
                cache[cache_key] = _evidence_mask(
                    variant.pipeline, sample, floor, domain
                )
            masks[(variant_index, sample_key)] = cache[cache_key]
    return masks


def _pairwise_iou(masks, domain):
    return np.array(
        [compare_masks(a, b, domain).iou for a, b in itertools.combinations(masks, 2)],
        dtype=np.float64,
    )


def _ensemble(cases, masks, pairwise=None):
    frequency, unstable = instability(masks)
    return EnsembleResult(
        cases=tuple(cases),
        selection_frequency=frequency,
        instability=unstable,
        pairwise_iou=(
            np.asarray(pairwise, dtype=np.float64)
            if pairwise is not None
            else np.empty(0, dtype=np.float64)
        ),
    )


def _channel_results(
    pipeline,
    variants,
    samples,
    masks,
    floor,
    domain,
    prune,
    locked_channels=(),
    cache=None,
):
    results = []
    evidence = pipeline.evidence_channels
    for channel in evidence:
        ablated = tuple(
            _Variant(
                variant.label, _without_channels(variant.pipeline, (channel.label,))
            )
            for variant in variants
        )
        ablated_masks = _compute_masks(ablated, samples, floor, domain, cache)
        deltas = [
            compare_masks(ablated_masks[key], masks[key], domain) for key in masks
        ]
        full_delta = compare_masks(
            ablated_masks[(0, "full")], masks[(0, "full")], domain
        )
        results.append(
            (
                channel.label,
                full_delta,
                sum(delta.changed_fraction > 0 for delta in deltas),
                max(delta.changed_fraction for delta in deltas),
                ablated_masks[(0, "full")] ^ masks[(0, "full")],
                all(delta.changed_fraction == 0 for delta in deltas),
            )
        )

    removed = []
    redundant_labels = {result[0] for result in results if result[-1]}
    current_variants = variants
    current_masks = masks
    locked_channels = set(locked_channels)
    if prune:
        for channel in reversed(evidence):
            if (
                channel.label in locked_channels
                or channel.label not in redundant_labels
            ):
                continue
            candidate_variants = tuple(
                _Variant(
                    variant.label, _without_channels(variant.pipeline, (channel.label,))
                )
                for variant in current_variants
            )
            candidate_masks = _compute_masks(
                candidate_variants, samples, floor, domain, cache
            )
            if all(
                np.array_equal(candidate_masks[key], current_masks[key])
                for key in current_masks
            ):
                removed.append(channel.label)
                current_variants = candidate_variants
                current_masks = candidate_masks

    removed_set = set(removed)
    channel_results = tuple(
        ChannelResult(
            label=label,
            full_delta=full_delta,
            cases_changed=changed,
            max_changed_fraction=maximum,
            contribution_mask=contribution,
            independently_redundant=redundant,
            removed=label in removed_set,
        )
        for label, full_delta, changed, maximum, contribution, redundant in results
    )
    return channel_results, tuple(reversed(removed))


def validate_mask(
    pipeline: Pipeline,
    run: int,
    selection: ShotSelection = BEAM_ON_SELECTION,
    design: Optional[MaskValidationDesign] = None,
    store: Optional[ImageStore] = None,
) -> MaskValidationReport:
    """Validate one run-local pipeline under declared data/model perturbations."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a Pipeline")
    if not isinstance(selection, ShotSelection):
        raise TypeError("selection must be a ShotSelection")
    design = design or MaskValidationDesign()
    if not isinstance(design, MaskValidationDesign):
        raise TypeError("design must be a MaskValidationDesign")
    store = store or ImageStore()

    variants = _variants(pipeline, design)
    sample = Sample.from_store(run, selection, pipeline.needs(), store=store)
    floor = np.asarray(pipeline.floor(sample))
    if floor.dtype != np.bool_ or floor.shape != sample.real.shape:
        raise TypeError("a pipeline floor must be boolean with the sample shape")
    domain = np.asarray(sample.real & ~floor, dtype=bool)
    if not domain.any():
        raise ValueError("mask validation domain is empty")

    samples, groups = _fold_samples(pipeline, run, selection, store, design, sample)
    mask_cache = {}
    masks = _compute_masks(variants, samples, floor, domain, mask_cache)
    baseline = masks[(0, "full")]

    data = {}
    for strategy, keys in groups.items():
        fold_masks = [masks[(0, key)] for key in keys]
        cases = [
            CaseDelta(f"fold {index}", compare_masks(mask, baseline, domain))
            for index, mask in enumerate(fold_masks)
        ]
        data[strategy] = _ensemble(cases, fold_masks, _pairwise_iou(fold_masks, domain))

    full_model_masks = [masks[(index, "full")] for index in range(len(variants))]
    model_cases = [
        CaseDelta(
            variants[index].label,
            compare_masks(full_model_masks[index], baseline, domain),
        )
        for index in range(1, len(variants))
    ]
    model = _ensemble(
        model_cases, full_model_masks, _pairwise_iou(full_model_masks, domain)
    )

    interaction = {}
    for strategy, keys in groups.items():
        ensemble_masks = [
            masks[(index, key)] for index in range(len(variants)) for key in keys
        ]
        cases = [
            CaseDelta(
                f"{variants[index].label} / fold {fold_index}",
                compare_masks(masks[(index, key)], masks[(0, key)], domain),
            )
            for index in range(1, len(variants))
            for fold_index, key in enumerate(keys)
        ]
        interaction[strategy] = _ensemble(cases, ensemble_masks)

    swept_channels = {
        sweep.path.split(".", 1)[0]
        for sweep in design.sweeps
        if not sweep.path.startswith("combiner.")
    }
    channels, removed = _channel_results(
        pipeline,
        variants,
        samples,
        masks,
        floor,
        domain,
        design.prune_redundant,
        swept_channels,
        mask_cache,
    )
    recommended = _without_channels(pipeline, removed)
    return MaskValidationReport(
        run=int(run),
        design=design,
        input_pipeline=pipeline,
        recommended_pipeline=recommended,
        sample=sample,
        floor=floor,
        domain=domain,
        baseline_mask=baseline,
        data=data,
        model=model,
        interaction=interaction,
        channels=channels,
        removed_channels=removed,
    )
