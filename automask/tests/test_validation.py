from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from automask.evaluation.metrics import compare_masks, instability
from automask.evaluation.validation import (
    MaskValidationDesign,
    ParameterSweep,
    _get_parameter,
    _set_parameter,
    validate_mask,
)
from automask.masking import Channel, Pipeline
from automask.shot_selection import ShotSelection
from automask.stats.base import STATS, StatSpec


def test_mask_delta_distinguishes_additions_removals_and_relocation():
    domain = np.ones((2, 3), dtype=bool)
    baseline = np.array([[True, False, False], [False, False, False]])
    candidate = np.array([[False, True, True], [False, False, False]])
    delta = compare_masks(candidate, baseline, domain)
    assert delta.iou == 0.0
    assert delta.changed_fraction == 3 / 6
    assert delta.added_fraction == 2 / 6
    assert delta.removed_fraction == 1 / 6
    assert delta.volume_delta == 1 / 6


def test_empty_masks_have_perfect_iou_and_no_change():
    empty = np.zeros((2, 2), dtype=bool)
    delta = compare_masks(empty, empty, np.ones_like(empty))
    assert delta.iou == 1.0
    assert delta.changed_fraction == 0.0


def test_instability_is_pairwise_disagreement_probability():
    masks = [
        np.array([[False, False, True]]),
        np.array([[False, True, True]]),
    ]
    frequency, unstable = instability(masks)
    np.testing.assert_array_equal(frequency, [[0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(unstable, [[0.0, 0.5, 0.0]])


@dataclass
class _Params:
    k: float = 0.5
    mode: str = "high"


@pytest.fixture
def test_stat(monkeypatch):
    spec = StatSpec(
        "validation_test_field",
        lambda sample, params: sample.mean,
        _Params,
        kind="field",
        mode="high",
        needs=("mean",),
    )
    monkeypatch.setitem(STATS, spec.name, spec)
    return spec.name


def _pipeline(test_stat, duplicate=False):
    channels = [Channel(test_stat, _Params(), field_reg=None, name="primary")]
    if duplicate:
        channels.append(Channel(test_stat, _Params(), field_reg=None, name="duplicate"))
    return Pipeline(channels)


def test_parameter_paths_clone_without_mutating_pipeline(test_stat):
    pipeline = _pipeline(test_stat)
    changed = _set_parameter(pipeline, "primary.params.k", 0.75)
    assert _get_parameter(pipeline, "primary.params.k") == 0.5
    assert _get_parameter(changed, "primary.params.k") == 0.75
    assert changed is not pipeline
    with pytest.raises(TypeError):
        _set_parameter(pipeline, "primary.params.k", "high")
    with pytest.raises(ValueError):
        _set_parameter(pipeline, "primary.params.unknown", 1.0)


def test_regularizer_parameter_paths():
    from automask.masking import production_pipeline

    pipeline = production_pipeline()
    changed = _set_parameter(pipeline, "variance.field_reg.tv.weight", 3.0)
    changed = _set_parameter(changed, "asic_polish.mask_reg.area_gate.min_area", 250)
    assert _get_parameter(changed, "variance.field_reg.tv.weight") == 3.0
    assert _get_parameter(changed, "asic_polish.mask_reg.area_gate.min_area") == 250
    assert _get_parameter(pipeline, "variance.field_reg.tv.weight") == 4.0


class _Store:
    def __init__(self):
        self.full = np.full((4, 4), 0.1, dtype=float)
        self.full[1, 1] = 1.0
        first = self.full.copy()
        second = self.full.copy()
        second[2, 2] = 1.0
        self.by_strategy = {
            "round_robin": (first, first),
            "chronological": (first, second),
        }

    def reduce(self, run, selection, reduction):
        assert reduction == "mean"
        return self.full

    def folds(self, run, selection, reduction, n_folds, strategy):
        assert reduction == "mean"
        assert n_folds == 2
        return self.by_strategy[strategy]


def test_validation_separates_data_model_and_interaction_axes(test_stat):
    pipeline = _pipeline(test_stat, duplicate=True)
    design = MaskValidationDesign(
        sweeps=(
            ParameterSweep(
                "primary.params.k", (0.25, 1.25), "declared threshold tolerance"
            ),
        ),
        n_folds=2,
    )
    report = validate_mask(pipeline, 999, ShotSelection(), design, store=_Store())

    assert report.data["round_robin"].pairwise_iou.tolist() == [1.0]
    assert report.data["chronological"].pairwise_iou.tolist() == [0.5]
    assert len(report.model.cases) == 2
    assert len(report.interaction["round_robin"].cases) == 4
    assert report.removed_channels == ()
    assert [
        channel.label for channel in report.recommended_pipeline.evidence_channels
    ] == ["primary", "duplicate"]
    assert "descriptive" in report.to_markdown()


def test_pruning_keeps_first_of_two_exact_duplicate_channels(test_stat):
    report = validate_mask(
        _pipeline(test_stat, duplicate=True),
        999,
        ShotSelection(),
        MaskValidationDesign(n_folds=2),
        store=_Store(),
    )
    assert report.removed_channels == ("duplicate",)
    assert [
        channel.label for channel in report.recommended_pipeline.evidence_channels
    ] == ["primary"]


def test_report_saves_structured_bundle(tmp_path, test_stat):
    report = validate_mask(
        _pipeline(test_stat),
        999,
        ShotSelection(),
        MaskValidationDesign(n_folds=2),
        store=_Store(),
    )
    output = report.save(tmp_path / "report")
    assert (output / "report.md").exists()
    assert (output / "metrics.json").exists()
    assert (output / "model_instability.npy").exists()
    assert (output / "overview.png").exists()
    with pytest.raises(FileExistsError):
        report.save(output)
