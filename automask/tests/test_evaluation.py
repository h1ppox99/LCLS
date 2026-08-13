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


def test_consistency_uses_fixed_folds_and_excludes_the_floor():
    class Pipeline:
        @staticmethod
        def needs():
            return ("mean",)

        @staticmethod
        def floor(sample):
            floor = np.zeros((3, 3), dtype=bool)
            floor[0, 0] = True
            return floor

        @staticmethod
        def run(sample):
            return Pipeline.floor(sample) | (sample.mean > 0)

    full = np.zeros((3, 3))
    full[1, 1] = 1
    late = np.zeros((3, 3))
    late[2, 2] = 1

    class Store:
        round_robin_data = tuple(full.copy() for _ in range(4))
        chronological_data = (
            full.copy(), full.copy(), late.copy(), late.copy())

        def reduce(self, run, selection, reduction):
            return self.round_robin_data[0]

        def folds(self, run, selection, reduction, n_folds, strategy):
            return (self.round_robin_data if strategy == "round_robin"
                    else self.chronological_data)

    result = evaluation.evaluate_consistency(
        Pipeline(), 999, store=Store(), n_folds=4)
    assert result["n_folds"] == 4
    assert result["round_robin"]["fold_iou"].shape == (6,)
    assert result["round_robin"]["fold_iou_mean"] == 1.0
    assert result["round_robin"]["fold_vs_full_iou_mean"] == 1.0
    assert result["chronological"]["fold_iou_mean"] == 1 / 3
