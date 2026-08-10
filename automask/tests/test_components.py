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
from automask.shot_selection import ShotMeta, ShotSelection
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
def _meta(n=100, beam=None, cc=None, vcc=None, monitor=None, run=475):
    """Synthetic ShotMeta: beam on, CC open, VCC closed, ramp intensity."""
    ramp = np.linspace(1.0, 100.0, n) if monitor is None else np.asarray(monitor)
    return ShotMeta(
        run=run,
        beam_on=np.ones(n, bool) if beam is None else np.asarray(beam),
        cc_open=np.ones(n, bool) if cc is None else np.asarray(cc),
        vcc_open=np.zeros(n, bool) if vcc is None else np.asarray(vcc),
        intensity={"sample_diode": ramp, "ipm2": ramp * 1000.0})


def test_shot_meta_is_built_from_canonical_profile_columns():
    profile = {
        "run": 12,
        "events": 3,
        "values": {
            "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]":
                np.array([1.0, 0.0, np.nan]),
            "ai/ch02": np.array([5.0, 0.0, np.nan]),
            "ai/ch03": np.array([0.0, 5.0, np.nan]),
            "diodeU/channels[0]": np.array([1.0, 2.0, 3.0]),
            "diodeU/sum": np.array([4.0, 5.0, 6.0]),
            "gas_detector/f_11_ENRC": np.array([7.0, 8.0, 9.0]),
        },
    }

    meta = ShotMeta.from_profile(profile)

    assert meta.run == 12
    np.testing.assert_array_equal(meta.beam_on, [True, False, False])
    np.testing.assert_array_equal(meta.cc_open, [True, False, False])
    np.testing.assert_array_equal(meta.vcc_open, [False, True, False])
    np.testing.assert_array_equal(meta.monitor("sample_diode"), [1.0, 2.0, 3.0])
    assert set(meta.intensity) == {"sample_diode", "diodeU", "gasdet"}


def test_feature_store_reuses_injected_shot_metadata(tmp_path):
    meta = _meta(run=12)
    store = FeatureStore(cache_dir=tmp_path, shot_meta=meta)

    assert store._meta(12) is meta
    try:
        store._meta(13)
    except ValueError as error:
        assert "run 12" in str(error) and "run 13" in str(error)
    else:
        raise AssertionError("expected a run mismatch to fail")


def test_beam_filter_selects_each_class():
    beam = np.array([True, True, False, False])
    meta = _meta(4, beam=beam, monitor=[1.0, 2.0, 3.0, 4.0])
    kw = dict(n_shots=None, filter_low=0.0, filter_high=0.0)
    assert list(ShotSelection(beam="on", **kw).resolve(meta)) == [0, 1]
    assert list(ShotSelection(beam="off", **kw).resolve(meta)) == [2, 3]
    assert list(ShotSelection(beam="any", **kw).resolve(meta)) == [0, 1, 2, 3]


def test_cc_and_vcc_are_independent_and_anded():
    cc = np.array([True, True, False, False])
    vcc = np.array([True, False, True, False])
    meta = _meta(4, cc=cc, vcc=vcc, monitor=[1.0, 2.0, 3.0, 4.0])
    kw = dict(n_shots=None, filter_low=0.0, filter_high=0.0)
    assert list(ShotSelection(cc="open", vcc="any", **kw).resolve(meta)) == [0, 1]
    assert list(ShotSelection(cc="any", vcc="open", **kw).resolve(meta)) == [0, 2]
    assert list(ShotSelection(cc="open", vcc="open", **kw).resolve(meta)) == [0]
    assert list(ShotSelection(cc="open", vcc="closed", **kw).resolve(meta)) == [1]
    assert list(ShotSelection(cc="closed", vcc="closed", **kw).resolve(meta)) == [3]
    assert list(ShotSelection(cc="any", vcc="any", **kw).resolve(meta)) == [0, 1, 2, 3]


def test_validity_floor_drops_zero_and_nan_readings():
    meta = _meta(4, monitor=[0.0, np.nan, 3.0, 4.0])
    sel = ShotSelection(n_shots=None, filter_low=0.0, filter_high=0.0)
    assert list(sel.resolve(meta)) == [2, 3]


def test_percentile_trim_drops_both_tails():
    meta = _meta(100)
    sel = ShotSelection(n_shots=None, filter_low=0.1, filter_high=0.1)
    kept = sel.resolve(meta)
    # ramp 1..100; np.quantile interpolates -> bounds 10.9 and 90.1
    assert kept.size == 80
    assert kept[0] == 10 and kept[-1] == 89


def test_n_shots_subsamples_evenly_across_survivors():
    meta = _meta(100)
    sel = ShotSelection(n_shots=10, filter_low=0.0, filter_high=0.0)
    kept = sel.resolve(meta)
    assert kept.size == 10
    assert kept[0] == 0 and kept[-1] == 99
    assert np.all(np.diff(kept) > 0)


def test_resolve_raises_when_nothing_matches():
    """Run 475 has zero VCC-open shots -- this must fail loudly, not silently."""
    meta = _meta(10, vcc=np.zeros(10, bool))
    try:
        ShotSelection(vcc="open").resolve(meta)
    except RuntimeError as e:
        assert "no shots match" in str(e)
        return
    raise AssertionError("expected RuntimeError for an empty selection")


def test_selection_rejects_unknown_monitors():
    for kwargs in ({"intensity": "nope"}, {"normalization": "nope"}):
        try:
            ShotSelection(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs!r}")


def test_selection_defaults_keep_both_local_runs_usable():
    """cc defaults to open (free: CC is open on 100% of shots in 389 and 475);
    vcc must default to any or run 475 yields nothing."""
    sel = ShotSelection()
    assert (sel.beam, sel.cc, sel.vcc) == ("on", "open", "any")
    assert sel.intensity == "sample_diode"


def test_shot_meta_rejects_ragged_and_unknown_monitors():
    for kwargs in ({"intensity": {"sample_diode": np.ones(5)}},   # wrong length
                   {"intensity": {"nope": np.ones(10)}},          # unknown monitor
                   {"intensity": {}}):                            # no monitor at all
        try:
            ShotMeta(run=1, beam_on=np.ones(10, bool), cc_open=np.ones(10, bool),
                     vcc_open=np.ones(10, bool), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs!r}")


def test_describe_breaks_down_each_filter():
    meta = _meta(10, beam=np.array([True] * 8 + [False] * 2),
                 vcc=np.array([True] * 3 + [False] * 7))
    d = ShotSelection(beam="on", cc="open", vcc="any").describe(meta)
    assert d["n_events"] == 10 and d["n_beam"] == 8
    assert d["n_cc"] == 10 and d["n_vcc"] == 10 and d["n_accessible"] == 8


def test_reference_intensity_uses_the_normalization_monitor():
    meta = _meta(100)
    sel = ShotSelection(intensity="sample_diode", normalization="ipm2")
    idx = sel.resolve(meta)
    assert np.isclose(sel.reference_intensity(meta, idx),
                      np.median(meta.intensity["ipm2"][idx]))


# -- FeatureSpec sources ---------------------------------------------------
def test_events_content_key_covers_the_whole_selection():
    """The warm XTC cache is keyed by this hash. It is deliberately NOT stable
    across ShotSelection field renames -- the CC/VCC rework invalidated it on
    purpose (see features/base.py)."""
    import hashlib
    import json
    from dataclasses import asdict

    spec = FeatureSpec("umean", "mean", ShotSelection(beam="on"))
    payload = json.dumps({"reduction": "mean", "selection": asdict(spec.selection)},
                         sort_keys=True)
    assert spec.content_key == hashlib.sha1(payload.encode()).hexdigest()[:12]
    other = FeatureSpec("umean", "mean", ShotSelection(beam="on", vcc="open"))
    assert spec.content_key != other.content_key


def test_calib_and_event_specs_do_not_collide():
    a = FeatureSpec("pedestal", source="calib", constant="pedestals", gain=0,
                    form="panel")
    b = FeatureSpec("pixel_rms", source="calib", constant="pixel_rms", gain=0,
                    form="panel")
    c = FeatureSpec("umean", "mean", ShotSelection(beam="on"))
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
