"""Concise, human-readable inspection of one experiment run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from automask.io.read_xtc import local_run_source
from automask.run_profile import RunProfile
from automask.utils import (
    detector_geometry,
    list_experiment_content,
    profile_run_values,
)


def _clean(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(rows, columns):
    if not rows:
        return "_None found._"
    headings = [heading for heading, _key in columns]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_clean(row.get(key, "")) for _heading, key in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


@dataclass
class RunInspectionReport:
    content: Dict[str, object]
    profile: RunProfile
    geometry: Dict[str, object]

    @property
    def run(self):
        return self.profile.run

    def to_markdown(self):
        varying = [row for row in self.profile.summary["xtc"] if not row["constant"]]
        constants = [row for row in self.profile.summary["xtc"] if row["constant"]]
        changing_epics = [
            row for row in self.profile.summary["epics"] if not row["constant"]
        ]
        constant_epics = sum(row["constant"] for row in self.profile.summary["epics"])
        summary_rows = [
            {"property": "Experiment", "value": self.content["experiment"]},
            {"property": "Run", "value": f"{self.run:04d}"},
            {"property": "Detector", "value": self.content["detector"]},
            {"property": "Detector source", "value": self.content["detector_source"]},
            {"property": "Decoded events", "value": f"{self.profile.events:,}"},
            {
                "property": "XTC files",
                "value": len(self.content["xtc"]["files"]),
            },
            {
                "property": "Calibration constants",
                "value": len(self.content["calibration"]),
            },
        ]
        geometry_rows = [
            {"property": key, "value": value} for key, value in self.geometry.items()
        ]
        if constant_epics == 1:
            hidden_epics = (
                " 1 constant EPICS variable is retained but not expanded below."
            )
        elif constant_epics:
            hidden_epics = (
                f" {constant_epics} constant EPICS variables are retained but "
                "not expanded below."
            )
        else:
            hidden_epics = ""
        lines = [
            f"# Run inspection — {self.content['experiment']} run {self.run:04d}",
            "",
            _table(summary_rows, (("property", "property"), ("value", "value"))),
            "",
            "## Relevant XTC files",
            "",
            _table(
                self.content["xtc"]["files"],
                (
                    ("file", "file"),
                    ("stream", "stream"),
                    ("chunk", "chunk"),
                    ("size", "size"),
                ),
            ),
            "",
            "## Applicable detector calibration files",
            "",
            _table(
                self.content["calibration"],
                (
                    ("constant", "constant"),
                    ("run range", "run range"),
                    ("size", "size"),
                ),
            ),
            "",
            "## Raw XTC payloads",
            "",
            _table(
                self.profile.payloads,
                (
                    ("source", "source"),
                    ("alias", "alias"),
                    ("type", "type"),
                    ("key", "key"),
                    ("coverage", "coverage"),
                    ("read errors", "read errors"),
                ),
            ),
            "",
            "## Detector geometry",
            "",
            _table(geometry_rows, (("property", "property"), ("value", "value"))),
            "",
            "## Profiled per-shot fields",
            "",
            "> `RunProfile` retains the complete event-aligned arrays and extraction "
            "metadata for every field; this report contains summaries only."
            + hidden_epics,
            "",
            "### Varying fields",
            "",
            _table(
                varying,
                (
                    ("field", "name"),
                    ("coverage", "coverage"),
                    ("summary", "summary"),
                ),
            ),
            "",
            "### Constant fields",
            "",
            _table(
                constants,
                (
                    ("field", "name"),
                    ("coverage", "coverage"),
                    ("summary", "summary"),
                ),
            ),
            "",
            "## Changing EPICS process variables",
            "",
            _table(
                changing_epics,
                (
                    ("alias", "alias"),
                    ("PV", "PV"),
                    ("coverage", "coverage"),
                    ("summary", "summary"),
                ),
            ),
        ]
        return "\n".join(lines)

    def display(self):
        markdown = self.to_markdown()
        try:
            from IPython.display import Markdown, display
        except ImportError:
            print(markdown)
            return
        display(Markdown(markdown))

    def save(self, directory, overwrite=False):
        path = Path(directory)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"report directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        return path


def inspect_run(
    experiment,
    run,
    detector,
    detector_source,
    detector_calib_type,
    source=None,
    detector_set=None,
    max_events=None,
):
    source = source or local_run_source(run)
    content = list_experiment_content(
        experiment, run, detector, detector_source, detector_calib_type
    )
    profile = profile_run_values(
        run,
        source=source,
        detector_set=detector_set,
        max_events=max_events,
    )
    geometry = detector_geometry(run, detector_name=detector, source=source)
    return RunInspectionReport(content=content, profile=profile, geometry=geometry)
