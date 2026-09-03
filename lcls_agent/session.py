"""In-memory automask session: live objects behind short string handles.

The agent (the model) cannot hold Python objects between tool calls, but the
host process that drives one agent run can. A :class:`Session` keeps the
expensive, stateful objects -- ``RunProfile``, ``ShotSelection``, ``Pipeline``
-- alive in memory and hands the agent a short handle (``prof-1``, ``sel-2``)
to name each one. Only the artifacts worth keeping are written to disk: the
run-profile cache (an expensive psana read), the inspection ``report.md``,
reduction/preview images, and the final mask deliverable. Selection and pipeline
parameters are never serialized -- they arrive inline as validated tool args and
live only as objects here.

This registry is an agent concept: it exists so the model can name objects across
tool calls (``lcls_agent.tools``). A human script has no need for handles -- it
holds the ``automask`` objects directly -- so this lives with the agent, not in
the ``automask`` library.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from automask.mask import Pipeline, production_pipeline
from automask.profiling.profile_store import ProfileStore
from automask.interface.recipes import (
    pipeline_from_dict,
    require_run_floor,
    selection_from_dict,
    validation_design_from_dict,
)
from automask.profiling.run_profile import RunProfile
from automask.selection.shot_selection import ShotSelection


class Session:
    """Registry of live automask objects addressed by short handles."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        # Created lazily by _artifact_dir, so a run that writes nothing (e.g. a
        # pure inspection question) leaves no empty directory behind.
        self.workdir = Path(workdir).expanduser().resolve()
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        # Profiles are cached across sessions by run, like ImageStore's reductions.
        self.profiles_store = ProfileStore()
        self._profiles: dict[str, RunProfile] = {}
        self._selections: dict[str, ShotSelection] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._counters: dict[str, int] = defaultdict(int)

    # -- handle bookkeeping -------------------------------------------------
    def _mint(self, kind: str) -> str:
        self._counters[kind] += 1
        return f"{kind}-{self._counters[kind]}"

    def _artifact_dir(self, handle: str) -> Path:
        target = self.workdir / handle
        target.mkdir(parents=True, exist_ok=True)
        return target

    def profile(self, handle: str) -> RunProfile:
        try:
            return self._profiles[handle]
        except KeyError:
            raise KeyError(
                f"unknown profile handle {handle!r}; known: {sorted(self._profiles)}"
            ) from None

    def selection(self, handle: str) -> ShotSelection:
        try:
            return self._selections[handle]
        except KeyError:
            raise KeyError(
                f"unknown selection handle {handle!r}; "
                f"known: {sorted(self._selections)}"
            ) from None

    def pipeline(self, handle: str) -> Pipeline:
        try:
            return self._pipelines[handle]
        except KeyError:
            raise KeyError(
                f"unknown pipeline handle {handle!r}; known: {sorted(self._pipelines)}"
            ) from None

    def register_profile(self, profile: RunProfile) -> str:
        handle = self._mint("prof")
        self._profiles[handle] = profile
        return handle

    def register_selection(self, selection: ShotSelection) -> str:
        handle = self._mint("sel")
        self._selections[handle] = selection
        return handle

    def register_pipeline(self, pipeline: Pipeline) -> str:
        handle = self._mint("pipe")
        self._pipelines[handle] = pipeline
        return handle

    # -- capabilities -------------------------------------------------------
    def catalog(self) -> dict:
        """Registered statistics, regularizers, and selection operators."""
        from automask.interface.catalog import capability_catalog

        return capability_catalog()

    # -- inspection ---------------------------------------------------------
    def inspect(
        self,
        run: int,
        *,
        experiment: str | None = None,
        detector: str | None = None,
        detector_source: str | None = None,
        detector_calib_type: str | None = None,
        max_events: int | None = None,
    ) -> dict:
        """Profile one run, cache the profile + report, return a handle."""
        from automask.io.read_xtc import (
            EXPERIMENT,
            JUNGFRAU_CALIB_TYPE,
            JUNGFRAU_NAME,
            JUNGFRAU_SOURCE,
        )
        from automask.profiling.run_inspection import inspect_run
        from automask.profiling.utils import configure_psana_environment

        configure_psana_environment()
        report = inspect_run(
            experiment or EXPERIMENT,
            run,
            detector or JUNGFRAU_NAME,
            detector_source or JUNGFRAU_SOURCE,
            detector_calib_type or JUNGFRAU_CALIB_TYPE,
            max_events=max_events,
        )
        handle = self.register_profile(report.profile)
        directory = self._artifact_dir(handle)
        (directory / "report.md").write_text(report.to_markdown(), encoding="utf-8")
        # A full profile is cached by run for later sessions; a bounded
        # (max_events) profile is development evidence, so it stays in memory only.
        cached = None
        if max_events is None:
            cached = str(self.profiles_store.save(report.profile))
        return {
            "profile": handle,
            "run": report.run,
            "events": report.profile.events,
            "fields": sorted(report.profile.values),
            "report": str(directory / "report.md"),
            "profile_cache": cached,
        }

    def load_profile(self, run: int) -> dict:
        """Reuse a run profile cached by an earlier session, skipping psana."""
        profile = self.profiles_store.load(int(run))
        handle = self.register_profile(profile)
        return {
            "profile": handle,
            "run": profile.run,
            "events": profile.events,
            "fields": sorted(profile.values),
            "source": str(self.profiles_store.path(int(run))),
        }

    # -- selection ----------------------------------------------------------
    def define_selection(self, spec: dict | ShotSelection | None = None) -> dict:
        """Register a shot selection from inline params; return its handle."""
        if isinstance(spec, ShotSelection):
            selection = spec
        else:
            selection = selection_from_dict(spec or {})
        handle = self.register_selection(selection)
        return {"selection": handle, "spec": _selection_summary(selection)}

    def describe_selection(self, profile: str, selection: str) -> dict:
        """Stage-by-stage shot counts for a selection against a profile."""
        prof = self.profile(profile)
        counts = self.selection(selection).describe(prof)
        return {"profile": profile, "selection": selection, "run": prof.run, **counts}

    # -- pipeline -----------------------------------------------------------
    def define_pipeline(self, spec: dict | Pipeline | None = None) -> dict:
        """Register a mask pipeline; default is the production recipe."""
        if isinstance(spec, Pipeline):
            pipeline = spec
        elif spec is None:
            pipeline = production_pipeline()
        else:
            pipeline = pipeline_from_dict(spec)
        handle = self.register_pipeline(pipeline)
        return {
            "pipeline": handle,
            "channels": [channel.label for channel in pipeline.channels],
            "needs": list(pipeline.needs()),
        }

    # -- preview ------------------------------------------------------------
    def preview_selection(
        self,
        profile: str,
        selection: str,
        *,
        reduction: str = "mean",
    ) -> dict:
        """Materialize and render one selected-shot reduction; persist images."""
        import matplotlib.pyplot as plt

        from automask.sample.image_store import ImageStore
        from automask.profiling.utils import configure_psana_environment
        from automask.viz import show

        configure_psana_environment()
        prof = self.profile(profile)
        sel = self.selection(selection)
        sel.resolve(prof)
        store = ImageStore(cache_dir=self.cache_dir, run_profile=prof)
        image = store.reduce(prof.run, sel, reduction)
        counts = store.counts(prof.run, sel, reduction) or sel.describe(prof)

        axis = show(image, title=f"Run {prof.run:04d}: selected-shot {reduction}")
        figure = axis.figure
        handle = self._mint("preview")
        directory = self._artifact_dir(handle)
        np.save(directory / f"{reduction}.npy", image)
        figure.savefig(directory / "preview.png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        return {
            "preview": handle,
            "run": prof.run,
            "reduction": reduction,
            "counts": counts,
            "image": _image_summary(image),
            "array": str(directory / f"{reduction}.npy"),
            "png": str(directory / "preview.png"),
        }

    # -- mask ---------------------------------------------------------------
    def build_mask(
        self,
        profile: str,
        selection: str,
        pipeline: str,
        *,
        gain: int = 0,
    ) -> dict:
        """Run a selection + pipeline into a final mask deliverable on disk."""
        import matplotlib.pyplot as plt

        from automask.sample.image_store import ImageStore
        from automask.sample import Sample
        from automask.mask.stats.base import STATS
        from automask.profiling.utils import configure_psana_environment
        from automask.viz import explain_panels, show, show_mask

        configure_psana_environment()
        prof = self.profile(profile)
        sel = self.selection(selection)
        pipe = self.pipeline(pipeline)
        require_run_floor(pipe)
        sel.resolve(prof)
        store = ImageStore(cache_dir=self.cache_dir, run_profile=prof)
        sample = Sample.from_store(prof.run, sel, pipe.needs(), store=store, gain=gain)
        floor = np.asarray(pipe.floor(sample))
        mask = np.asarray(pipe.run(sample, floor=floor))
        if mask.dtype != np.bool_ or mask.shape != sample.real.shape:
            raise TypeError("pipeline must return a boolean mask with the sample shape")
        if np.any(floor & ~mask):
            raise ValueError("final mask does not contain the required floor")

        layer_records = [
            {
                "channel": channel.label,
                "stat": channel.stat,
                "kind": STATS[channel.stat].kind,
                "pixels": int(np.asarray(channel.pick(sample), dtype=bool).sum()),
            }
            for channel in pipe.channels
        ]

        figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        show(sample.mean, ax=axes[0], cbar=False, title="selected-shot mean")
        show_mask(mask, ax=axes[1], title=f"mask: {100 * mask.mean():.2f}% of canvas")
        figure.suptitle(f"run {prof.run:04d}")
        figure.tight_layout(rect=[0, 0, 1, 0.95])

        handle = self._mint("mask")
        directory = self._artifact_dir(handle)
        np.save(directory / "mask.npy", mask)
        np.save(directory / "floor.npy", floor)
        figure.savefig(directory / "overview.png", dpi=120, bbox_inches="tight")
        plt.close(figure)

        explain_records = []
        panels = pipe.explain(sample)
        if panels:
            for name, panel in panels.items():
                record = {"panel": name, "mask": bool(panel.mask)}
                if not panel.mask:
                    safe = name.replace(" — ", "_").replace(" ", "_")
                    path = directory / f"explain_{safe}.npy"
                    np.save(path, np.asarray(panel.array, dtype=np.float32))
                    record["array"] = str(path)
                explain_records.append(record)
            fig_explain = explain_panels(pipe, sample, panels=panels)
            fig_explain.savefig(directory / "explain.png", dpi=120, bbox_inches="tight")
            plt.close(fig_explain)

        real = np.asarray(sample.real, dtype=bool)
        return {
            "mask": handle,
            "run": prof.run,
            "needs": list(pipe.needs()),
            "selected_shots": sel.describe(prof)["n_selected"],
            "canvas_pixels": int(mask.size),
            "real_pixels": int(real.sum()),
            "floor_pixels": int(floor.sum()),
            "masked_pixels": int(mask.sum()),
            "masked_fraction_canvas": float(mask.mean()),
            "evidence_pixels_outside_floor": int(np.count_nonzero(mask & ~floor)),
            "layers": layer_records,
            "mask_array": str(directory / "mask.npy"),
            "overview": str(directory / "overview.png"),
            "explain": explain_records,
            "explain_overview": (
                str(directory / "explain.png") if explain_records else None
            ),
        }

    # -- validation ---------------------------------------------------------
    def validate_mask(
        self,
        profile: str,
        selection: str,
        pipeline: str,
        *,
        design: dict | None = None,
    ) -> dict:
        """Run declared data/model perturbations; persist the report."""
        from automask.evaluation import MaskValidationDesign, validate_mask
        from automask.sample.image_store import ImageStore
        from automask.profiling.utils import configure_psana_environment

        configure_psana_environment()
        prof = self.profile(profile)
        sel = self.selection(selection)
        pipe = self.pipeline(pipeline)
        require_run_floor(pipe)
        spec = (
            MaskValidationDesign()
            if design is None
            else validation_design_from_dict(design)
        )
        sel.resolve(prof)
        store = ImageStore(cache_dir=self.cache_dir, run_profile=prof)
        report = validate_mask(pipe, prof.run, selection=sel, design=spec, store=store)
        handle = self._mint("validation")
        directory = self._artifact_dir(handle)
        report.save(directory / "report")
        recommended = self.register_pipeline(report.recommended_pipeline)
        return {
            "validation": handle,
            "run": prof.run,
            "report": str(directory / "report" / "report.md"),
            "recommended_pipeline": recommended,
        }


def _selection_summary(selection: ShotSelection) -> dict:
    return {
        "where": [condition.label() for condition in selection.where],
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


def _image_summary(image: np.ndarray) -> dict:
    values = np.asarray(image)
    finite = np.isfinite(values)
    observed = values[finite]
    percentiles = (
        {}
        if not observed.size
        else {
            str(percentile): float(np.percentile(observed, percentile))
            for percentile in (1, 5, 50, 95, 99)
        }
    )
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_pixels": int(finite.sum()),
        "zero_pixels": int(np.count_nonzero(finite & (values == 0))),
        "percentiles": percentiles,
    }
