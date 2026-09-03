import numpy as np
import pytest

from automask.profiling.profile_store import ProfileStore
from automask.profiling.run_profile import RunProfile


def _profile():
    return RunProfile(
        run=12,
        events=5,
        payloads=[{"source": "test"}],
        values={
            "intensity": np.asarray([1.0, 2.0, np.nan, 4.0, 5.0]),
            "state": np.asarray(["dark", "lit", None, "lit", b"lit"], dtype=object),
        },
        summary={"xtc": [], "epics": []},
        epics=[],
    )


def test_round_trip_is_pickle_free(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    original = _profile()

    directory = store.save(original)
    restored = store.load(12)

    assert restored.run == 12
    assert restored.events == 5
    np.testing.assert_equal(restored.values["intensity"], original.values["intensity"])
    assert restored.values["state"].tolist() == original.values["state"].tolist()
    # Object column stored as integer codes, and the archive loads without pickle.
    with np.load(directory / "values.npz", allow_pickle=False) as archive:
        assert archive.files == ["column_0000", "column_0001"]
        assert archive["column_0001"].dtype.kind in "iu"
        np.testing.assert_array_equal(archive["column_0001"], [0, 1, 2, 1, 3])


def test_has_and_try_load(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    assert not store.has(12)
    assert store.try_load(12) is None

    store.save(_profile())

    assert store.has(12)
    assert store.try_load(12).run == 12


def test_load_missing_run_is_a_clear_error(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    with pytest.raises(FileNotFoundError, match="no cached profile for run 0099"):
        store.load(99)
