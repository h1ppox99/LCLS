"""Typed, human-readable output for run-local mask validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from automask.evaluation.metrics import MaskDelta
from automask.interface.recipes import pipeline_to_dict


@dataclass(frozen=True)
class CaseDelta:
    label: str
    delta: MaskDelta


@dataclass
class EnsembleResult:
    cases: Tuple[CaseDelta, ...]
    selection_frequency: np.ndarray
    instability: np.ndarray
    pairwise_iou: np.ndarray


@dataclass(frozen=True)
class ChannelResult:
    label: str
    full_delta: MaskDelta
    cases_changed: int
    max_changed_fraction: float
    contribution_mask: np.ndarray
    independently_redundant: bool
    removed: bool


def _plain(value):
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pipeline_config(pipeline) -> dict:
    return pipeline_to_dict(pipeline)


def _ensemble_dict(result: EnsembleResult) -> dict:
    return {
        "cases": [
            {"label": case.label, **case.delta.as_dict()} for case in result.cases
        ],
        "pairwise_iou": result.pairwise_iou.tolist(),
        "selection_frequency_min": float(result.selection_frequency.min()),
        "selection_frequency_max": float(result.selection_frequency.max()),
        "instability_max": float(result.instability.max()),
    }


@dataclass
class MaskValidationReport:
    run: int
    design: object
    input_pipeline: object
    recommended_pipeline: object
    sample: object
    floor: np.ndarray
    domain: np.ndarray
    baseline_mask: np.ndarray
    data: Dict[str, EnsembleResult]
    model: EnsembleResult
    interaction: Dict[str, EnsembleResult]
    channels: Tuple[ChannelResult, ...]
    removed_channels: Tuple[str, ...]

    @property
    def evidence_fraction(self) -> float:
        return float(self.baseline_mask.sum()) / int(self.domain.sum())

    def to_dict(self) -> dict:
        return {
            "run": self.run,
            "design": _plain(self.design.as_dict()),
            "evidence_fraction": self.evidence_fraction,
            "removed_channels": list(self.removed_channels),
            "input_pipeline": _pipeline_config(self.input_pipeline),
            "recommended_pipeline": _pipeline_config(self.recommended_pipeline),
            "data": {key: _ensemble_dict(value) for key, value in self.data.items()},
            "model": _ensemble_dict(self.model),
            "interaction": {
                key: _ensemble_dict(value) for key, value in self.interaction.items()
            },
            "channels": [
                {
                    "label": channel.label,
                    **channel.full_delta.as_dict(),
                    "cases_changed": channel.cases_changed,
                    "max_changed_fraction": channel.max_changed_fraction,
                    "independently_redundant": channel.independently_redundant,
                    "removed": channel.removed,
                }
                for channel in self.channels
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Mask validation — run {self.run:04d}",
            "",
            f"Baseline evidence masks **{100 * self.evidence_fraction:.3f}%** "
            "of valid non-floor pixels.",
        ]
        if self.design.sweeps:
            lines += [
                "",
                "## Declared model perturbations",
                "",
                "| parameter | values | reason |",
                "| --- | --- | --- |",
            ]
            for sweep in self.design.sweeps:
                values = ", ".join(repr(value) for value in sweep.values)
                reason = sweep.reason.replace("|", "\\|")
                lines.append(f"| `{sweep.path}` | `{values}` | {reason} |")
        lines += [
            "",
            "## Data perturbations",
            "",
            "Pairwise IoU is the size-invariant reproducibility of the masked set across "
            "folds (`1 − IoU` ≈ the fraction that is noise); read mean **and** worst case, "
            "since one low-IoU fold can flag false positives the mean hides.",
            "",
            "| strategy | folds | pairwise IoU (mean / min) | vs-full IoU (mean / min) | "
            "changed area (mean / max) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for strategy, result in self.data.items():
            versus = [case.delta for case in result.cases]
            pw = np.asarray(result.pairwise_iou, dtype=float)
            vf = np.array([d.iou for d in versus], dtype=float)
            ch = np.array([d.changed_fraction for d in versus], dtype=float)
            pw_mean, pw_min = (
                (float(pw.mean()), float(pw.min())) if pw.size else (1.0, 1.0)
            )
            vf_mean, vf_min = (
                (float(vf.mean()), float(vf.min())) if vf.size else (1.0, 1.0)
            )
            ch_mean, ch_max = (
                (float(ch.mean()), float(ch.max())) if ch.size else (0.0, 0.0)
            )
            lines.append(
                f"| {strategy} | {len(versus)} | "
                f"{pw_mean:.4f} / {pw_min:.4f} | "
                f"{vf_mean:.4f} / {vf_min:.4f} | "
                f"{100 * ch_mean:.4f}% / {100 * ch_max:.4f}% |"
            )
        # Name the least-reproducible fold per strategy so a consistently-deviant fold
        # is visible without opening metrics.json. Most meaningful for `chronological`,
        # where a fold is a time window: a real transient worth excluding, not a random
        # split as in `round_robin`.
        worst = [
            (strategy, min(result.cases, key=lambda c: c.delta.iou))
            for strategy, result in self.data.items()
            if result.cases
        ]
        if worst:
            lines += [
                "",
                "Most-deviant fold (investigate for a transient, "
                "especially `chronological`):",
                "",
            ]
            lines += [
                f"- **{strategy}**: `{case.label}` — vs-full IoU {case.delta.iou:.4f}, "
                f"changed {100 * case.delta.changed_fraction:.4f}%"
                for strategy, case in worst
            ]

        lines += [
            "",
            "## Model perturbations",
            "",
            "| variant | IoU | changed | added | removed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        if self.model.cases:
            for case in self.model.cases:
                delta = case.delta
                lines.append(
                    f"| `{case.label}` | {delta.iou:.4f} | "
                    f"{100 * delta.changed_fraction:.4f}% | "
                    f"{100 * delta.added_fraction:.4f}% | "
                    f"{100 * delta.removed_fraction:.4f}% |"
                )
        else:
            lines.append("| _no model perturbations declared_ | — | — | — | — |")

        lines += [
            "",
            "## Data–model interaction",
            "",
            "| strategy | comparisons | minimum IoU | maximum changed area |",
            "| --- | ---: | ---: | ---: |",
        ]
        for strategy, result in self.interaction.items():
            deltas = [case.delta for case in result.cases]
            minimum = min((delta.iou for delta in deltas), default=1.0)
            maximum = max((delta.changed_fraction for delta in deltas), default=0.0)
            lines.append(
                f"| {strategy} | {len(deltas)} | {minimum:.4f} | {100 * maximum:.4f}% |"
            )

        lines += [
            "",
            "## Channel ablations",
            "",
            "| channel | cases changed | maximum changed area | action |",
            "| --- | ---: | ---: | --- |",
        ]
        for channel in self.channels:
            if channel.removed:
                action = "removed: exactly redundant"
            elif channel.independently_redundant:
                action = "retained: sweep target or sequential dependency"
            else:
                action = "retained"
            lines.append(
                f"| `{channel.label}` | {channel.cases_changed} | "
                f"{100 * channel.max_changed_fraction:.4f}% | {action} |"
            )

        lines += [
            "",
            "## Interpretation limits",
            "",
            "This report is descriptive. Stability and exact redundancy do not prove "
            "that an unlabelled mask is scientifically correct. No confidence "
            "intervals, correctness oracle, or pass/fail thresholds were applied.",
        ]
        return "\n".join(lines)

    def figures(self) -> Dict[str, object]:
        import matplotlib.pyplot as plt

        from automask.viz import channel_panels, show, show_mask

        figures = {}
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        show(self.sample.mean, ax=axes[0], cbar=False, title="selected-shot mean")
        show_mask(self.baseline_mask, ax=axes[1], title="baseline evidence mask")
        fig.suptitle(f"run {self.run:04d} — validation baseline")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figures["overview"] = fig

        maps = [(f"data: {key}", value.instability) for key, value in self.data.items()]
        maps.append(("model", self.model.instability))
        maps.extend(
            (f"interaction: {key}", value.instability)
            for key, value in self.interaction.items()
        )
        fig, axes = plt.subplots(
            1, len(maps), figsize=(4.2 * len(maps), 4.3), squeeze=False
        )
        image = None
        for ax, (label, values) in zip(axes[0], maps):
            image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=0.5)
            ax.set_title(label)
            ax.axis("off")
        if image is not None:
            fig.colorbar(
                image, ax=list(axes[0]), fraction=0.025, pad=0.02, label="2p(1-p)"
            )
        fig.suptitle("instability maps")
        figures["instability"] = fig

        if self.channels:
            fig, axes = plt.subplots(
                1,
                len(self.channels),
                figsize=(4.2 * len(self.channels), 4.3),
                squeeze=False,
            )
            for ax, channel in zip(axes[0], self.channels):
                show_mask(channel.contribution_mask, ax=ax, title=channel.label)
            fig.suptitle("full-sample leave-one-channel-out changes")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            figures["ablations"] = fig
            figures["channels"] = channel_panels(
                self.input_pipeline, self.sample, floor_row=True
            )
        return figures

    def display(self):
        markdown = self.to_markdown()
        figures = self.figures()
        try:
            from IPython.display import Markdown, display
        except ImportError:
            print(markdown)
            return figures
        display(Markdown(markdown))
        for figure in figures.values():
            display(figure)
        return figures

    def save(self, directory, overwrite: bool = False) -> Path:
        path = Path(directory)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"report directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.md").write_text(self.to_markdown())
        (path / "metrics.json").write_text(json.dumps(self.to_dict(), indent=2))
        np.save(path / "baseline_mask.npy", self.baseline_mask)
        np.save(path / "floor.npy", self.floor)
        for axis, result in [
            *((f"data_{key}", value) for key, value in self.data.items()),
            ("model", self.model),
            *((f"interaction_{key}", value) for key, value in self.interaction.items()),
        ]:
            np.save(
                path / f"{axis}_selection_frequency.npy", result.selection_frequency
            )
            np.save(path / f"{axis}_instability.npy", result.instability)
        import matplotlib.pyplot as plt

        for name, figure in self.figures().items():
            figure.savefig(path / f"{name}.png", dpi=120, bbox_inches="tight")
            plt.close(figure)
        return path
