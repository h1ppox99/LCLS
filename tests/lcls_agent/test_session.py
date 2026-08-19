"""The agent's handle registry: define/describe objects without psana."""

import numpy as np
import pytest

from automask.recipes import require_run_floor
from automask.run_profile import RunProfile
from lcls_agent.session import Session


def _profile():
    return RunProfile(
        run=12,
        events=5,
        payloads=[{"source": "test"}],
        values={"intensity": np.asarray([1.0, 2.0, np.nan, 4.0, 5.0])},
        summary={"xtc": [], "epics": []},
        epics=[],
    )


def test_selection_handles_and_counts(tmp_path):
    session = Session(tmp_path / "work")
    profile = session.register_profile(_profile())
    selection = session.define_selection(
        {
            "where": [{"field": "intensity", "operator": ">=", "value": 2.0}],
            "n_shots": 2,
        }
    )["selection"]

    counts = session.describe_selection(profile, selection)

    assert counts["run"] == 12
    assert counts["n_selected"] == 2


def test_empty_selection_reports_zero_without_raising(tmp_path):
    session = Session(tmp_path / "work")
    profile = session.register_profile(_profile())
    selection = session.define_selection(
        {"where": [{"field": "intensity", "operator": ">", "value": 100.0}]}
    )["selection"]

    assert session.describe_selection(profile, selection)["n_selected"] == 0


def test_unknown_handle_is_a_clear_error(tmp_path):
    session = Session(tmp_path / "work")
    with pytest.raises(KeyError, match="unknown profile handle"):
        session.profile("prof-999")


def test_handles_increment_per_kind(tmp_path):
    session = Session(tmp_path / "work")
    a = session.define_selection({})["selection"]
    b = session.define_selection({})["selection"]
    assert (a, b) == ("sel-1", "sel-2")


def test_default_pipeline_carries_the_run_floor(tmp_path):
    session = Session(tmp_path / "work")
    handle = session.define_pipeline()["pipeline"]
    require_run_floor(session.pipeline(handle))
