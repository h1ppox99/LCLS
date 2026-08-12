"""Ground-truth-free evaluation of a masking procedure on one run."""
from __future__ import annotations

from typing import Optional

import numpy as np

from automask.evaluation import resampling
from automask.evaluation.azimuthal import azimuthal_diagnostics, build_frame
from automask.evaluation.schemas import Estimate, RuntimeEvaluation
from automask.evaluation.stability import sampling_stability, temporal_stability
from automask.image_store import ImageStore
from automask.sample import Sample
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection


def _estimate(values) -> Estimate:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("an evaluation estimate has no finite observations")
    spread = float(values.std(ddof=1)) if values.size > 1 else None
    return Estimate(
        value=float(values.mean()), standard_deviation=spread, n=int(values.size)
    )


def evaluate_runtime(
    pipeline,
    run: int,
    selection: ShotSelection = BEAM_ON_SELECTION,
    store: Optional[ImageStore] = None,
    seed: int = 0,
    resamples: int = 20,
    controls: int = 20,
) -> RuntimeEvaluation:
    """Measure stability and physical consistency without reading a reference mask."""
    sample = Sample.from_store(int(run), selection, pipeline.needs(), store=store)
    floor = np.asarray(pipeline.floor(sample), dtype=bool)
    mask = np.asarray(pipeline.run(sample))
    if mask.dtype != np.bool_ or mask.shape != floor.shape:
        raise TypeError("a pipeline must return a boolean mask with the floor shape")

    moments = resampling.load(int(run), selection=selection)
    seed_sequence = np.random.SeedSequence([int(seed), int(run)])
    sampling_seed, control_seed = seed_sequence.spawn(2)
    sampling = sampling_stability(
        pipeline, sample, moments, np.random.default_rng(sampling_seed), resamples)
    temporal = temporal_stability(pipeline, sample, moments)

    frame = build_frame(sample, floor)
    azimuthal = azimuthal_diagnostics(
        frame, mask, np.random.default_rng(control_seed), controls)

    return RuntimeEvaluation(
        run=int(run),
        masked_fraction=float(mask.mean()),
        floor_contained=bool(np.all(mask[floor])),
        sampling_stability=_estimate(sampling),
        temporal_stability=_estimate([temporal]),
        azimuthal_excess=_estimate([azimuthal["excess"]]),
        azimuthal_gain=_estimate(azimuthal["gain"]),
        azimuthal_win_rate=_estimate(azimuthal["win_rate"]),
    )
