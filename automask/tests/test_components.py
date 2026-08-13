"""
Tests for the pipeline components added alongside the pedestal detector. Plain
assert-style functions so they run either under pytest or directly (pytest is
not installed in the ana env):

    python -m automask.tests.test_components

Self-contained: tiny arrays only, no frozen dataset and no psana.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from automask.image_store import (
    ImageStore, _calibration_content_key, _content_key, _reduction_stub,
)
from automask.masking import Channel, Pipeline, production_pipeline
from automask.regularization.area_gate import area_gate
from automask.regularization.blob_scale import blob_scale
from automask.regularization.fill_holes import fill_holes
from automask.run_profile import RunProfile
from automask.shot_selection import Condition, PercentileTrim, ShotSelection
from automask.stats.asic_polish import median_polish


# -- regularizers ----------------------------------------------------------
def test_area_gate_drops_small_components():
    m = np.zeros((40, 40), bool)
    m[2:4, 2:4] = True            # 4 px
    m[10:20, 10:20] = True        # 100 px
    out = area_gate(m, min_area=50)
    assert out[10:20, 10:20].all()
    assert not out[2:4, 2:4].any()
    assert out.sum() == 100


def test_fill_holes_fills_interior_but_does_not_grow():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    m[9:11, 9:11] = False         # punch an interior hole
    out = fill_holes(m)
    assert out[9:11, 9:11].all()
    assert out.sum() == 100       # the 10x10 square, nothing more
    assert not out[4, 4]          # boundary unchanged, unlike `pad`


def test_blob_scale_preserves_unit_variance_on_white_noise():
    """The renormalization is what lets `k` stay in sigma after aggregation.

    Asserted on the INTERIOR: `mode="nearest"` replicates values past the edge,
    which correlates the noise there and inflates the variance within ~R of the
    border (see the note in blob_scale's docstring)."""
    z = np.random.default_rng(0).normal(size=(400, 400))
    for radius in (4, 8, 16, 24):
        out = blob_scale(z, radii=(radius,))
        b = 2 * radius
        assert abs(out[b:-b, b:-b].std() - 1.0) < 0.05, (radius, out.std())


def test_blob_scale_amplifies_a_coherent_patch():
    """A weak per-pixel excess becomes strongly detectable once aggregated --
    the whole reason the stat/regularizer split exists."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(200, 200))
    yy, xx = np.mgrid[0:200, 0:200]
    patch = np.hypot(yy - 100, xx - 100) <= 12
    z[patch] += 1.5                       # far below any sensible per-pixel k
    out = blob_scale(z, radii=(12,))
    assert z[patch].max() < 6.0
    assert out[patch].max() > 12.0


def test_blob_scale_max_over_scales_is_monotone():
    z = np.random.default_rng(2).normal(size=(120, 120))
    single = blob_scale(z, radii=(8,))
    multi = blob_scale(z, radii=(4, 8, 16))
    assert (multi >= single - 1e-9).all()


# -- median polish ---------------------------------------------------------
def test_median_polish_removes_row_and_column_structure():
    rng = np.random.default_rng(3)
    rows = rng.normal(scale=50, size=(64, 1))
    cols = rng.normal(scale=50, size=(1, 64))
    block = rows + cols + rng.normal(size=(64, 64))
    res = median_polish(block)
    assert res.std() < 1.5                # striping gone, noise left


def test_median_polish_keeps_a_compact_anomaly():
    """A defect covering a minority of rows/columns must survive the fit --
    this is why the model is robust rather than least-squares."""
    rng = np.random.default_rng(4)
    block = rng.normal(size=(64, 64)) + np.arange(64)[None, :] * 3.0
    block[20:28, 20:28] += 40.0
    res = median_polish(block)
    assert res[20:28, 20:28].mean() > 30.0


# -- Channel regularizer composition ---------------------------------------
def test_stages_normalizes_all_accepted_shapes():
    assert Channel._stages(None, None) == []
    assert Channel._stages("tv", None) == [("tv", None)]
    assert Channel._stages(["a", "b"], None) == [("a", None), ("b", None)]
    assert Channel._stages(["a", "b"], [1, 2]) == [("a", 1), ("b", 2)]


def test_stages_rejects_mismatched_params():
    for names, params in ((["a", "b"], [1]), (["a", "b"], 1)):
        try:
            Channel._stages(names, params)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {names!r}, {params!r}")


def test_channel_rejects_wrong_regularizer_kind():
    """A mask regularizer in a field slot would silently mangle the field."""
    try:
        Channel("variance", field_reg="area_gate")
    except ValueError as e:
        assert "kind=" in str(e)
        return
    raise AssertionError("expected ValueError for a mask reg in a field slot")


def test_channel_rejects_field_reg_on_a_pick_stat():
    for field_reg in ("tv", ["tv", "blob_scale"]):
        try:
            Channel("hough_lines", field_reg=field_reg)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for field_reg={field_reg!r}")


def test_pipeline_collects_every_needed_array():
    pipeline = production_pipeline()
    assert pipeline.needs() == ("mean", "pedestals", "real", "status_as_mask", "std")
    assert [c.label for c in pipeline.floor_channels] == ["geometry", "status_as_mask"]
    blackhat = Pipeline([Channel("blackhat")])
    assert blackhat.needs() == ("mean", "real")
    assert blackhat.floor_channels == []


def test_pipeline_rejects_duplicate_channel_labels():
    """Two channels keyed the same would silently overwrite each other's evidence."""
    try:
        Pipeline([Channel("blackhat"), Channel("blackhat")])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("expected a duplicate-label error")
    named = Pipeline([Channel("blackhat"), Channel("blackhat", name="wide")])
    assert [c.label for c in named.channels] == ["blackhat", "wide"]


def test_floor_channel_is_not_gated_by_real(monkeypatch):
    """A dead pixel reads zero, so gating the floor by `real` would erase exactly
    the pixels the floor exists to mask."""
    import automask.stats.status_as_mask as status_stat
    from automask.sample import Sample
    from automask.stats.status_as_mask import StatusAsMaskParams

    monkeypatch.setattr(status_stat, "panel_to_asm", lambda panel, run: panel[0])
    mean = np.ones((5, 5))
    mean[2, 2] = 0.0                      # dead pixel: no value, so not `real`
    status = np.ones((1, 5, 5), dtype=np.uint8)
    status[0, 2, 2] = 0                   # psana: 0 == bad
    sample = Sample(run=475, arrays={"mean": mean, "status_as_mask": status})

    assert not sample.real[2, 2]
    floor = Channel("status_as_mask", StatusAsMaskParams(pad=0),
                    field_reg=None).pick(sample)
    assert floor[2, 2]
    assert floor.sum() == 1


# -- shot selection --------------------------------------------------------
def _profile(n=100, **values):
    columns = {"intensity": np.linspace(1.0, 100.0, n), **values}
    return RunProfile(475, n, [], columns, {}, [])


def test_run_profile_returns_canonical_columns():
    profile = _profile(3, state=np.array(["a", "b", "a"]))
    np.testing.assert_array_equal(profile.column("state"), ["a", "b", "a"])
    try:
        profile.column("missing")
    except KeyError as error:
        assert "available examples" in str(error)
    else:
        raise AssertionError("expected a missing-field error")


def test_image_store_reuses_injected_run_profile(tmp_path):
    profile = _profile(3)
    store = ImageStore(cache_dir=tmp_path, run_profile=profile)
    assert store.profile(475) is profile


def test_image_store_profiles_each_run_once(tmp_path, monkeypatch):
    import automask.utils as utils

    calls = []

    def profile_run(run, show):
        calls.append((run, show))
        return RunProfile(run, 0, [], {}, {}, [])

    monkeypatch.setattr(utils, "profile_run_values", profile_run)
    store = ImageStore(cache_dir=tmp_path)
    assert store.profile(12) is store.profile(12)
    assert store.profile(13) is store.profile(13)
    assert calls == [(12, False), (13, False)]


def test_conditions_are_field_native_and_anded():
    profile = _profile(
        5,
        branch=np.array([0, 0, 1, 1, 1]),
        quality=np.array([0.1, 0.5, 0.4, 0.8, 0.2]),
    )
    selection = ShotSelection(where=(
        Condition("branch", "==", 1),
        Condition("quality", "between", (0.3, 0.8)),
    ))
    np.testing.assert_array_equal(selection.resolve(profile), [2, 3])


def test_conditions_support_categorical_values():
    profile = _profile(4, mode=np.array(["sample", "dark", "sample", "calib"]))
    selection = ShotSelection(where=(
        Condition("mode", "in", ("dark", "calib")),
    ))
    np.testing.assert_array_equal(selection.resolve(profile), [1, 3])


def test_percentile_trim_drops_both_tails_after_conditions():
    profile = _profile(100, admitted=np.r_[np.zeros(10), np.ones(90)])
    selection = ShotSelection(
        where=(Condition("admitted", "==", 1),),
        trim=PercentileTrim("intensity", low=0.1, high=0.1),
    )
    kept = selection.resolve(profile)
    assert kept.size == 72
    assert kept[0] == 19 and kept[-1] == 90


def test_n_shots_subsamples_evenly_across_survivors():
    kept = ShotSelection(n_shots=10).resolve(_profile(100))
    assert kept.size == 10 and kept[0] == 0 and kept[-1] == 99
    assert np.all(np.diff(kept) > 0)


def test_neutral_selection_needs_no_fields():
    profile = RunProfile(12, 3, [], {}, {}, [])
    np.testing.assert_array_equal(ShotSelection().resolve(profile), [0, 1, 2])


def test_normalization_excludes_unusable_values():
    profile = _profile(4, monitor=np.array([1.0, 0.0, np.nan, 4.0]))
    selection = ShotSelection(normalization="monitor")
    indices = selection.resolve(profile)
    np.testing.assert_array_equal(indices, [0, 3])
    assert selection.normalization_reference(profile, indices) == 2.5


def test_selection_reports_empty_and_missing_fields():
    profile = _profile(4, state=np.zeros(4))
    for selection, error_type in (
        (ShotSelection(where=(Condition("state", ">", 0),)), RuntimeError),
        (ShotSelection(where=(Condition("missing", "==", 1),)), KeyError),
        (ShotSelection(trim=PercentileTrim("missing")), KeyError),
    ):
        try:
            selection.resolve(profile)
        except error_type:
            continue
        raise AssertionError(f"expected {error_type.__name__} for {selection}")


def test_describe_reports_generic_selection_stages():
    profile = _profile(10, state=np.array([1] * 8 + [0] * 2))
    selection = ShotSelection(
        where=(Condition("state", "==", 1),), n_shots=3,
    )
    description = selection.describe(profile)
    assert description["n_events"] == 10
    assert description["conditions"][0]["n_matching"] == 8
    assert description["n_after_conditions"] == 8
    assert description["n_eligible"] == 8
    assert description["n_after_trim"] == 8
    assert description["n_selected"] == 3


def test_describe_reports_an_empty_selection_without_raising():
    profile = _profile(4, state=np.zeros(4))
    description = ShotSelection(where=(
        Condition("state", ">", 0),
    )).describe(profile)
    assert description["n_eligible"] == 0
    assert description["n_selected"] == 0


# -- ImageStore ------------------------------------------------------------
def test_reduction_content_key_covers_the_whole_selection():
    """The warm XTC cache hash covers the complete generic selection recipe."""
    import hashlib
    import json
    from dataclasses import asdict

    selection = ShotSelection(where=(
        Condition("state", "==", "sample"),
    ))
    payload = json.dumps({"reduction": "mean", "selection": asdict(selection)},
                         sort_keys=True)
    assert _content_key(selection, "mean") == hashlib.sha1(
        payload.encode()
    ).hexdigest()[:12]
    assert _reduction_stub(475, selection, "mean") == \
        "mean_b7a9d7729a67_run0475"
    variants = (
        ShotSelection(where=(Condition("state", "==", "dark"),)),
        ShotSelection(where=selection.where, trim=PercentileTrim("intensity", 0.1, 0.2)),
        ShotSelection(where=selection.where, n_shots=10),
        ShotSelection(where=selection.where, normalization="intensity"),
    )
    keys = {_content_key(selection, "mean"), _content_key(selection, "std")}
    keys.update(_content_key(other, "mean") for other in variants)
    assert len(keys) == 2 + len(variants)


def test_calibration_and_reduction_keys_do_not_collide():
    selection = ShotSelection(where=(
        Condition("state", "==", "sample"),
    ))
    keys = {
        _calibration_content_key("pedestals", 0),
        _calibration_content_key("rms", 0),
        _content_key(selection, "mean"),
    }
    assert len(keys) == 3
    assert _reduction_stub(475, selection, "mean").startswith("mean_")
    assert _calibration_content_key("pedestals", 0) == "194fcda7cea5"


def test_image_store_validates_reduction_form_and_gain(tmp_path):
    store = ImageStore(cache_dir=tmp_path)
    selection = ShotSelection()
    calls = (
        lambda: store.reduce(1, selection, "sum"),
        lambda: store.reduce(1, selection, "mean", form="bad"),
        lambda: store.reduce(1, object(), "mean"),
        lambda: store.calibration(1, "pedestals", gain=-1),
        lambda: store.calibration(1, "", gain=0),
    )
    for call in calls:
        try:
            call()
        except (TypeError, ValueError):
            continue
        raise AssertionError("expected validation error")


def _small_store(tmp_path, monkeypatch):
    import automask.image_store as image_store
    import automask.io.read_xtc as read_xtc

    monkeypatch.setattr(
        read_xtc, "panel_geometry",
        lambda run, source=None: (
            np.array([[[0, 0], [1, 1]]]), np.array([[[0, 1], [0, 1]]])
        ),
    )
    profile = RunProfile(7, 2, [], {}, {}, [], source=object())
    return ImageStore(cache_dir=tmp_path, run_profile=profile)


def test_mean_std_are_co_computed_and_cache_hits_skip_compute(tmp_path, monkeypatch):
    store = _small_store(tmp_path, monkeypatch)
    calls = []

    def accumulate(run, selection):
        calls.append((run, selection))
        return (
            np.arange(4).reshape(1, 2, 2),
            np.arange(4, 8).reshape(1, 2, 2),
            {"n_events": 2, "n_eligible": 2, "n_selected": 2, "n_used": 2},
        )

    monkeypatch.setattr(store, "_accumulate", accumulate)
    selection = ShotSelection()
    np.testing.assert_array_equal(store.reduce(7, selection, "mean"), [[0, 1], [2, 3]])
    np.testing.assert_array_equal(
        store.reduce(7, selection, "std", form="panel"),
        np.arange(4, 8, dtype=np.float32).reshape(1, 2, 2),
    )
    assert len(calls) == 1
    assert store.counts(7, selection, "mean")["n_used"] == 2


def test_median_mad_are_co_computed(tmp_path, monkeypatch):
    store = _small_store(tmp_path, monkeypatch)
    calls = []

    def stage(run, selection, path):
        path.touch()
        calls.append((run, selection))
        return {"n_events": 2, "n_eligible": 2, "n_selected": 2, "n_used": 2}

    monkeypatch.setattr(store, "_stage_frames", stage)
    monkeypatch.setattr(
        store, "_robust_reduce",
        lambda path: (np.ones((1, 2, 2)), np.full((1, 2, 2), 2.0)),
    )
    selection = ShotSelection()
    assert np.all(store.reduce(7, selection, "mad") == 2.0)
    assert np.all(store.reduce(7, selection, "median", form="panel") == 1.0)
    assert len(calls) == 1


def test_image_store_caches_fixed_folds_and_halves(tmp_path, monkeypatch):
    import automask.io.read_xtc as read_xtc

    profile = RunProfile(7, 20, [], {}, {}, [], source=object())
    store = ImageStore(cache_dir=tmp_path, run_profile=profile)
    calls = []

    def frames(run, indices, source=None):
        calls.append(run)
        for index in indices:
            yield int(index), np.full((1, 2, 2), float(index))

    monkeypatch.setattr(read_xtc, "iter_calibrated", frames)
    monkeypatch.setattr(
        read_xtc, "panel_geometry",
        lambda run, source=None: (
            np.array([[[0, 0], [1, 1]]]), np.array([[[0, 1], [0, 1]]])
        ),
    )
    selection = ShotSelection()
    folds = store.folds(7, selection, "mean", n_folds=4)
    halves = store.halves(7, selection, "std", n_folds=4)

    assert len(folds) == 4
    np.testing.assert_allclose([fold[0, 0] for fold in folds], np.arange(4) + 8)
    np.testing.assert_allclose([half[0, 0] for half in halves], np.sqrt(8.25))
    np.testing.assert_allclose(store.reduce(7, selection, "mean"), 9.5)
    assert any("fold-03-of-04" in path.name for path in tmp_path.iterdir())
    assert calls == [7]


def test_image_store_builds_robust_fold_reductions(tmp_path, monkeypatch):
    import automask.io.read_xtc as read_xtc

    profile = RunProfile(7, 20, [], {}, {}, [], source=object())
    store = ImageStore(cache_dir=tmp_path, run_profile=profile)

    def frames(run, indices, source=None):
        for index in indices:
            yield int(index), np.full((1, 2, 2), float(index))

    monkeypatch.setattr(read_xtc, "iter_calibrated", frames)
    monkeypatch.setattr(
        read_xtc, "panel_geometry",
        lambda run, source=None: (
            np.array([[[0, 0], [1, 1]]]), np.array([[[0, 1], [0, 1]]])
        ),
    )
    folds = store.folds(7, ShotSelection(), "median", n_folds=5)
    halves = store.halves(7, ShotSelection(), "mad", n_folds=5)

    np.testing.assert_allclose([fold[0, 0] for fold in folds], np.arange(5) + 7.5)
    np.testing.assert_allclose([half[0, 0] for half in halves], 1.4826 * 2.5)


def test_calibration_is_cached_in_panel_form_only(tmp_path, monkeypatch):
    """Assembling scatters onto a zero canvas, which would read as "bad" over
    every unmapped pixel of a status mask -- so the store never assembles one."""
    import automask.io.read_xtc as read_xtc

    store = _small_store(tmp_path, monkeypatch)
    calls = []

    def calibration(run, constant, gain=0, source=None):
        calls.append((run, constant, gain, source))
        return np.arange(12).reshape(3, 1, 2, 2)[gain]

    monkeypatch.setattr(read_xtc, "detector_calibration", calibration)
    panel = store.calibration(7, "pedestals", gain=1)
    np.testing.assert_array_equal(panel, np.arange(4, 8).reshape(1, 2, 2))
    np.testing.assert_array_equal(store.calibration(7, "pedestals", gain=1), panel)
    assert len(calls) == 1
    assert not list(tmp_path.glob("*_asm.npy"))
    assert len(calls) == 1


def test_xtc_run_discovery_uses_the_filesystem(tmp_path):
    from automask.io.read_xtc import available_xtc_runs

    for name in (
        "xppl1016922-r0475-s01-c00.xtc",
        "xppl1016922-r0378-s00-c00.xtc",
        "xppl1016922-r0475-s00-c00.xtc",
        "another-r0999-s00-c00.xtc",
        "xppl1016922-r0396-not-an-xtc-name.xtc",
    ):
        (tmp_path / name).touch()
    assert available_xtc_runs(tmp_path) == (378, 475)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:                       # noqa: BLE001 - test runner
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed} passed" + (f", {failed} failed" if failed else ""))
    raise SystemExit(1 if failed else 0)


def test_hydra_production_experiment_matches_the_python_recipe():
    """conf/experiment/production.yaml duplicates production_pipeline() as data.

    A silent divergence means the documented sweep command runs a different mask
    than the library does, which had already happened once (the yaml named
    sigma_clipping long after the Python recipe moved on).
    """
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    conf_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
    from automask.studies.sweep_hyperparameters import build_pipeline

    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment=production"])
        from_yaml = build_pipeline(cfg)
    from_python = production_pipeline("union")

    assert [c.label for c in from_yaml.channels] == \
        [c.label for c in from_python.channels]
    assert from_yaml.needs() == from_python.needs()
