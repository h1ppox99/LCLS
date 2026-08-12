from __future__ import annotations

import numpy as np

import automask.evaluation as evaluation


def test_xtc_runs_are_partitioned_between_fit_and_validation():
    assert evaluation.FIT_RUNS + evaluation.VALIDATION_RUNS == evaluation.ALL_RUNS
    assert set(evaluation.FIT_RUNS).isdisjoint(evaluation.VALIDATION_RUNS)
    assert evaluation.FIT_RUNS and evaluation.VALIDATION_RUNS


def test_labelled_evaluation_defaults_to_validation_runs(monkeypatch):
    import automask.evaluation.labelled as labelled

    class Pipeline:
        @staticmethod
        def needs():
            return ()

        @staticmethod
        def floor(sample):
            return np.zeros((2, 2), dtype=bool)

        @staticmethod
        def run(sample):
            return np.array([[True, False], [False, False]])

    class Sample:
        def __init__(self, run):
            self.run = run

    monkeypatch.setattr(
        labelled.Sample, "from_store",
        lambda run, selection, needs, store=None: Sample(run),
    )
    monkeypatch.setattr(
        labelled, "reference_mask",
        lambda run: np.array([[True, False], [False, False]]),
    )

    result = evaluation.evaluate(Pipeline())
    assert tuple(key for key in result if isinstance(key, int)) == evaluation.VALIDATION_RUNS
    assert result["mean"]["residual_iou"] == 1.0


def test_runtime_evaluation_is_label_free(monkeypatch):
    import automask.evaluation.runtime as runtime

    class Pipeline:
        @staticmethod
        def needs():
            return ()

        @staticmethod
        def floor(sample):
            return np.array([[True, False], [False, False]])

        @staticmethod
        def run(sample):
            return np.array([[True, False], [False, False]])

    sample = type("Sample", (), {"run": 999})()
    monkeypatch.setattr(
        runtime.Sample, "from_store",
        lambda run, selection, needs, store=None: sample,
    )
    monkeypatch.setattr(runtime.resampling, "load", lambda run, selection: object())
    monkeypatch.setattr(runtime, "sampling_stability", lambda *args: np.array([0.9, 1.0]))
    monkeypatch.setattr(runtime, "temporal_stability", lambda *args: 0.8)
    monkeypatch.setattr(runtime, "build_frame", lambda sample, floor: object())
    monkeypatch.setattr(
        runtime, "azimuthal_diagnostics",
        lambda frame, mask, rng, controls: {
            "excess": 0.2,
            "gain": np.array([0.1, 0.2]),
            "win_rate": np.array([0.6, 0.7]),
        },
    )
    monkeypatch.setattr(
        evaluation, "reference_mask",
        lambda run: (_ for _ in ()).throw(AssertionError("ground truth was read")),
    )
    result = evaluation.evaluate_runtime(Pipeline(), 999, resamples=2, controls=2)
    assert result.run == 999
    assert result.floor_contained
    assert result.sampling_stability.value == 0.95
