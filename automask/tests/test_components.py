"""
Tests for the pipeline components added alongside the pedestal detector. Plain
assert-style functions so they run either under pytest or directly (pytest is
not installed in the ana env):

    python -m automask.tests.test_components

Self-contained: tiny arrays only, no frozen dataset and no psana.
"""
from __future__ import annotations

import numpy as np

from automask.features.base import FeatureSpec
from automask.features.store import FeatureStore
from automask.masking import Detector
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


# -- Detector regularizer composition --------------------------------------
def test_stages_normalizes_all_accepted_shapes():
    assert Detector._stages(None, None) == []
    assert Detector._stages("tv", None) == [("tv", None)]
    assert Detector._stages(["a", "b"], None) == [("a", None), ("b", None)]
    assert Detector._stages(["a", "b"], [1, 2]) == [("a", 1), ("b", 2)]


def test_stages_rejects_mismatched_params():
    for names, params in ((["a", "b"], [1]), (["a", "b"], 1)):
        try:
            Detector._stages(names, params)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {names!r}, {params!r}")


def test_detector_rejects_wrong_regularizer_kind():
    """A mask regularizer in a field slot would silently mangle the field."""
    try:
        Detector("variance", field_reg="area_gate")
    except ValueError as e:
        assert "kind=" in str(e)
        return
    raise AssertionError("expected ValueError for a mask reg in a field slot")


def test_detector_rejects_field_reg_on_a_pick_stat():
    for field_reg in ("tv", ["tv", "blob_scale"]):
        try:
            Detector("hough_lines", field_reg=field_reg)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for field_reg={field_reg!r}")


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


def test_feature_store_reuses_injected_run_profile(tmp_path):
    profile = _profile(3)
    store = FeatureStore(cache_dir=tmp_path, run_profile=profile)
    assert store.profile(475) is profile


def test_feature_store_profiles_each_run_once(tmp_path, monkeypatch):
    import automask.utils as utils

    calls = []

    def profile_run(run, show):
        calls.append((run, show))
        return RunProfile(run, 0, [], {}, {}, [])

    monkeypatch.setattr(utils, "profile_run_values", profile_run)
    store = FeatureStore(cache_dir=tmp_path)
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


# -- FeatureSpec sources ---------------------------------------------------
def test_events_content_key_covers_the_whole_selection():
    """The warm XTC cache hash covers the complete generic selection recipe."""
    import hashlib
    import json
    from dataclasses import asdict

    spec = FeatureSpec("umean", "mean", ShotSelection(where=(
        Condition("state", "==", "sample"),
    )))
    payload = json.dumps({"reduction": "mean", "selection": asdict(spec.selection)},
                         sort_keys=True)
    assert spec.content_key == hashlib.sha1(payload.encode()).hexdigest()[:12]
    other = FeatureSpec("umean", "mean", ShotSelection(where=(
        Condition("state", "==", "dark"),
    )))
    assert spec.content_key != other.content_key


def test_calib_and_event_specs_do_not_collide():
    a = FeatureSpec("pedestal", source="calib", constant="pedestals", gain=0,
                    form="panel")
    b = FeatureSpec("pixel_rms", source="calib", constant="pixel_rms", gain=0,
                    form="panel")
    c = FeatureSpec("umean", "mean", ShotSelection(where=(
        Condition("state", "==", "sample"),
    )))
    keys = {a.content_key, b.content_key, c.content_key}
    assert len(keys) == 3
    assert a.cache_stub(475).startswith("pedestalsg0_")


def test_feature_spec_validates_its_source():
    for kwargs in ({"source": "calib"},                    # missing constant/gain
                   {"reduction": "mean"},                  # events, no selection
                   {"source": "nope"}):
        try:
            FeatureSpec("x", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs!r}")


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
