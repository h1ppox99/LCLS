from __future__ import annotations

import json

import pytest

from agent.entrypoint import center_outcome
from pipeline.lab6_ring_validator import load_pyfai_lab6_model
from pipeline.step2c_estimate_center import (
    TAN_RATIO_110_100,
    UNINDEXED_ANCHORS_PX,
    _verify_fit,
)


def _fit_inputs(*, separation: float, span: float, ratio_error: float = 0.2):
    independent = {
        "100": {"center": [990.0, 35.0], "angular_span_deg": 90.0},
        "110": {"center": [990.0 + separation, 35.0], "angular_span_deg": span},
    }
    joint = {
        "ratio_error_pct": ratio_error,
        "per_ring_residual": {
            "100": {"residual_mad_px": 1.0},
            "110": {"residual_mad_px": 1.0},
        },
    }
    proposal = {"pair_evidence": [{"contrast": 0.3}, {"contrast": 0.4}]}
    return independent, joint, proposal


def test_only_indexed_reflections_constrain_center():
    assert UNINDEXED_ANCHORS_PX == [367, 531, 734]
    assert pytest.approx(1.47122, rel=1e-5) == TAN_RATIO_110_100


def test_pyfai_supplies_lab6_ring_order_relationships():
    model = load_pyfai_lab6_model()
    assert model["provider"] == "pyFAI"
    assert model["calibrant"] == "LaB6_SRM660c"
    assert [ring["hkl"] for ring in model["rings"][:5]] == ["100", "110", "111", "200", "210"]
    assert model["rings"][1]["radius_ratio_to_100"] == pytest.approx(1.471228, rel=1e-5)


def test_all_ring_failure_rejects_semantic_assignment():
    inputs = _fit_inputs(separation=2.0, span=45.0)
    ring_validation = {
        "ring_model": {"provider": "pyFAI", "calibrant": "LaB6_SRM660c"},
        "n_in_field": 6,
        "n_matched": 2,
        "failed_hkl": ["110", "111", "210", "211"],
        "all_in_field_match": False,
    }
    verdict = _verify_fit(*inputs, ring_validation)
    assert verdict["verdict"] == "revise"
    assert verdict["failure_type"] == "ring_semantics_failure"


def test_short_outer_arc_is_uncertainty_not_escalation():
    inputs = _fit_inputs(separation=3.0, span=15.0)
    verdict = _verify_fit(*inputs)
    assert verdict["verdict"] == "accept"


def test_well_covered_inconsistent_centers_escalate():
    inputs = _fit_inputs(separation=12.0, span=45.0)
    verdict = _verify_fit(*inputs)
    assert verdict["verdict"] == "escalate"
    assert verdict["failure_type"] == "identifiability_failure"


def test_consistent_two_ring_fit_is_accepted():
    inputs = _fit_inputs(separation=2.0, span=45.0)
    verdict = _verify_fit(*inputs)
    assert verdict["verdict"] == "accept"


@pytest.mark.parametrize(
    ("decision", "center", "expected"),
    [
        ("reuse", {"row": 992.0, "col": 35.0}, "accept"),
        ("estimate", {"row": 993.0, "col": 34.0}, "accept"),
        ("revise", None, "revise"),
        ("escalate", None, "escalate"),
    ],
)
def test_host_respects_center_terminal_decision(tmp_path, decision, center, expected):
    artifact = {
        "coordinate_system": "assembled_row_col",
        "decision": decision,
        "center": center,
    }
    path = tmp_path / "image_center.json"
    path.write_text(json.dumps(artifact))
    assert center_outcome(path) == expected
