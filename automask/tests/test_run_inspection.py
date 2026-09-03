import numpy as np

import automask.profiling.run_inspection as run_inspection
from automask.profiling.run_profile import RunProfile


def _report():
    profile = RunProfile(
        run=12,
        events=4,
        payloads=[
            {
                "source": "DetInfo(NoDetector.0:Evr.0)",
                "alias": "evr0",
                "type": "EvrData.DataV4",
                "key": "—",
                "coverage": "100.0%",
                "read errors": 0,
            }
        ],
        values={
            "ai/ch03": np.asarray([0.0, 0.0, 5.0, 5.0]),
            "ai/ch02": np.zeros(4),
            "EPICS/delay": np.ones(4),
        },
        summary={
            "xtc": [
                {
                    "name": "ai/ch03",
                    "source": "ai",
                    "field": "ch03",
                    "type": "smalldata_tools",
                    "origin": "smalldata_tools",
                    "coverage": "100.0%",
                    "summary": "binary; 1 changes; 0 x 2, 5 x 2",
                    "constant": False,
                },
                {
                    "name": "ai/ch02",
                    "source": "ai",
                    "field": "ch02",
                    "type": "smalldata_tools",
                    "origin": "smalldata_tools",
                    "coverage": "100.0%",
                    "summary": "constant: 0",
                    "constant": True,
                },
            ],
            "epics": [
                {
                    "name": "EPICS/delay",
                    "alias": "delay",
                    "PV": "XPP:DELAY",
                    "dtype": "float64",
                    "coverage": "100.0%",
                    "summary": "constant: 1",
                    "constant": True,
                }
            ],
        },
        epics=[],
    )
    content = {
        "experiment": "xpptest",
        "run": 12,
        "detector": "jungfrau",
        "detector_source": "DetInfo(Jungfrau)",
        "detector_calib_type": "Jungfrau::CalibV1",
        "xtc": {
            "files": [
                {
                    "file": "xpptest-r0012-s00-c00.xtc",
                    "stream": "s00",
                    "chunk": "c00",
                    "size": "1.00 GiB",
                    "path": "/private/data/run.xtc",
                }
            ],
            "total_bytes": 1,
        },
        "calibration": [
            {
                "constant": "pedestals",
                "run range": "10–end",
                "size": "2.00 MiB",
                "path": "/private/calib/pedestals.data",
            }
        ],
    }
    return run_inspection.RunInspectionReport(
        content=content,
        profile=profile,
        geometry={"native shape": (2, 512, 1024)},
    )


def test_run_inspection_markdown_is_concise_and_actionable():
    markdown = _report().to_markdown()

    assert "# Run inspection — xpptest run 0012" in markdown
    assert "## Raw XTC payloads" in markdown
    assert "EvrData.DataV4" in markdown
    assert "ai/ch03" in markdown
    assert "binary; 1 changes; 0 x 2, 5 x 2" in markdown
    assert "ai/ch02" in markdown
    assert "1 constant EPICS variable is retained" in markdown
    assert "XPP:DELAY" not in markdown
    assert "/private/" not in markdown


def test_run_inspection_save_writes_the_displayed_markdown(tmp_path):
    report = _report()

    output = report.save(tmp_path / "inspection")

    assert (output / "report.md").read_text(encoding="utf-8") == report.to_markdown()


def test_inspect_run_builds_one_report(monkeypatch):
    expected = _report()
    source = object()
    calls = {}

    monkeypatch.setattr(
        run_inspection,
        "list_experiment_content",
        lambda *args: expected.content,
    )

    def profile(run, **kwargs):
        calls["profile"] = (run, kwargs)
        return expected.profile

    def geometry(run, **kwargs):
        calls["geometry"] = (run, kwargs)
        return expected.geometry

    monkeypatch.setattr(run_inspection, "profile_run_values", profile)
    monkeypatch.setattr(run_inspection, "detector_geometry", geometry)

    report = run_inspection.inspect_run(
        "xpptest",
        12,
        "jungfrau",
        "DetInfo(Jungfrau)",
        "Jungfrau::CalibV1",
        source=source,
    )

    assert report.profile is expected.profile
    assert calls["profile"] == (
        12,
        {"source": source, "detector_set": None, "max_events": None},
    )
    assert calls["geometry"] == (
        12,
        {"detector_name": "jungfrau", "source": source},
    )
