"""Bidirectional dict<->object conversion for automask selection/pipeline params.

Parameters reach automask inline -- as tool args validated by the SDK schema, or
as ``--*-json`` on the CLI -- never as standalone recipe files. This module is the
one place those field-native dicts become ``ShotSelection``/``Pipeline`` objects
(``*_from_dict``, the validated door in) and back (``*_to_dict``, for echoing a
spec and for the catalog's parameter examples).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from automask.masking import Channel, Pipeline
from automask.regularization.base import REGULARIZERS
from automask.shot_selection import Condition, PercentileTrim, ShotSelection
from automask.stats.base import STATS


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping(value: Any, context: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return dict(value)


def _keys(value: dict, allowed: set[str], required: set[str], context: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{context} is missing {sorted(missing)}")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)}")


def _params(registry: dict, name: str, value: Any, context: str):
    if name not in registry:
        raise ValueError(f"unknown {context} {name!r}; choose from {sorted(registry)}")
    payload = {} if value is None else _mapping(value, f"{context} params")
    params_type = registry[name].params
    allowed = {item.name for item in fields(params_type)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"{context} {name!r} has unknown params {sorted(unknown)}; "
            f"choose from {sorted(allowed)}"
        )
    try:
        return params_type(**payload)
    except (TypeError, ValueError) as error:
        raise type(error)(f"invalid {context} {name!r} params: {error}") from error


def selection_to_dict(selection: ShotSelection) -> dict:
    return {
        "where": [
            {
                "field": condition.field,
                "operator": condition.operator,
                "value": to_plain(condition.value),
            }
            for condition in selection.where
        ],
        "trim": (
            None
            if selection.trim is None
            else {
                "field": selection.trim.field,
                "low": selection.trim.low,
                "high": selection.trim.high,
            }
        ),
        "n_shots": selection.n_shots,
        "normalization": selection.normalization,
    }


def selection_from_dict(value: Any) -> ShotSelection:
    payload = _mapping(value, "selection")
    _keys(
        payload,
        {"where", "trim", "n_shots", "normalization"},
        set(),
        "selection",
    )
    raw_conditions = payload.get("where", [])
    if not isinstance(raw_conditions, list):
        raise TypeError("selection.where must be a JSON array")
    conditions = []
    for index, raw in enumerate(raw_conditions):
        item = _mapping(raw, f"selection.where[{index}]")
        _keys(
            item,
            {"field", "operator", "value"},
            {"field", "operator"},
            f"selection.where[{index}]",
        )
        conditions.append(Condition(item["field"], item["operator"], item.get("value")))

    raw_trim = payload.get("trim")
    trim = None
    if raw_trim is not None:
        trim_payload = _mapping(raw_trim, "selection.trim")
        _keys(
            trim_payload,
            {"field", "low", "high"},
            {"field"},
            "selection.trim",
        )
        trim = PercentileTrim(
            trim_payload["field"],
            low=trim_payload.get("low", 0.0),
            high=trim_payload.get("high", 0.0),
        )
    return ShotSelection(
        where=tuple(conditions),
        trim=trim,
        n_shots=payload.get("n_shots"),
        normalization=payload.get("normalization"),
    )


def _stages_to_dict(channel: Channel, kind: str) -> list[dict]:
    names = getattr(channel, kind)
    values = getattr(channel, f"{kind}_params")
    return [
        {
            "name": name,
            "params": to_plain(
                params if params is not None else REGULARIZERS[name].params()
            ),
        }
        for name, params in channel._stages(names, values)
    ]


def pipeline_to_dict(pipeline: Pipeline) -> dict:
    return {
        "channels": [
            {
                "name": channel.label,
                "stat": channel.stat,
                "params": to_plain(channel._params()),
                "field_regularizers": _stages_to_dict(channel, "field_reg"),
                "mask_regularizers": _stages_to_dict(channel, "mask_reg"),
            }
            for channel in pipeline.channels
        ],
    }


def _stages_from_dict(value: Any, context: str):
    if value is None:
        value = []
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    names = []
    params = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{context}[{index}]")
        _keys(
            item,
            {"name", "params"},
            {"name"},
            f"{context}[{index}]",
        )
        name = item["name"]
        names.append(name)
        params.append(_params(REGULARIZERS, name, item.get("params"), "regularizer"))
    if not names:
        return None, None
    if len(names) == 1:
        return names[0], params[0]
    return names, params


def pipeline_from_dict(value: Any) -> Pipeline:
    payload = _mapping(value, "pipeline")
    _keys(
        payload,
        {"channels"},
        {"channels"},
        "pipeline",
    )
    if not isinstance(payload["channels"], list):
        raise TypeError("pipeline.channels must be a JSON array")
    channels = []
    for index, raw in enumerate(payload["channels"]):
        context = f"pipeline.channels[{index}]"
        item = _mapping(raw, context)
        _keys(
            item,
            {
                "name",
                "stat",
                "params",
                "field_regularizers",
                "mask_regularizers",
            },
            {"stat"},
            context,
        )
        stat = item["stat"]
        stat_params = _params(STATS, stat, item.get("params"), "statistic")
        field_reg, field_reg_params = _stages_from_dict(
            item.get("field_regularizers"), f"{context}.field_regularizers"
        )
        mask_reg, mask_reg_params = _stages_from_dict(
            item.get("mask_regularizers"), f"{context}.mask_regularizers"
        )
        channels.append(
            Channel(
                stat=stat,
                params=stat_params,
                field_reg=field_reg,
                field_reg_params=field_reg_params,
                mask_reg=mask_reg,
                mask_reg_params=mask_reg_params,
                name=item.get("name"),
            )
        )
    return Pipeline(channels)


def require_run_floor(pipeline: Pipeline) -> None:
    required = {name for name, spec in STATS.items() if spec.kind == "floor"}
    present = {channel.stat for channel in pipeline.floor_channels}
    missing = required - present
    if missing:
        raise ValueError(
            "run-level masking requires the registered floor channels; "
            f"missing {sorted(missing)}"
        )


def validation_design_to_dict(design) -> dict:
    return to_plain(design.as_dict())


def validation_design_from_dict(value: Any):
    from automask.evaluation import MaskValidationDesign, ParameterSweep

    payload = _mapping(value, "validation design")
    _keys(
        payload,
        {"sweeps", "n_folds", "strategies", "prune_redundant"},
        set(),
        "validation design",
    )
    sweeps = []
    for index, raw in enumerate(payload.get("sweeps", [])):
        item = _mapping(raw, f"validation design.sweeps[{index}]")
        _keys(
            item,
            {"path", "values", "reason"},
            {"path", "values", "reason"},
            f"validation design.sweeps[{index}]",
        )
        sweeps.append(
            ParameterSweep(item["path"], tuple(item["values"]), item["reason"])
        )
    return MaskValidationDesign(
        sweeps=tuple(sweeps),
        n_folds=payload.get("n_folds", 10),
        strategies=tuple(payload.get("strategies", ("round_robin", "chronological"))),
        prune_redundant=payload.get("prune_redundant", True),
    )
