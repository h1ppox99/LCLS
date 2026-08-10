import sys
from types import SimpleNamespace

import numpy as np
import pytest

from automask.io.psana1 import Psana1RunSource
from automask.io.read_xtc import detector_calibration


class FakeDetector:
    def __init__(self, name):
        self.name = name

    def pedestals(self, run):
        return np.full((3, 2, 2, 2), run, dtype=np.float32)

    def rms(self, run):
        return np.full((3, 2, 2, 2), run / 10, dtype=np.float32)


class FakePsana:
    def __init__(self):
        self.options = []
        self.calls = []

    def setOption(self, name, value):
        self.options.append((name, value))

    def DataSource(self, *args):
        self.calls.append(("serial", args))
        return SimpleNamespace(kind="serial", args=args)

    def MPIDataSource(self, *args):
        self.calls.append(("mpi", args))
        return SimpleNamespace(kind="mpi", args=args)

    def Detector(self, name):
        return FakeDetector(name)


def test_explicit_file_source_opens_with_local_calibration(tmp_path, monkeypatch):
    first = tmp_path / "run-s00.xtc"
    second = tmp_path / "run-s01.xtc"
    first.touch()
    second.touch()
    calibration = tmp_path / "calib"
    calibration.mkdir()
    psana = FakePsana()
    monkeypatch.setitem(sys.modules, "psana", psana)

    source = Psana1RunSource.from_files(
        "xpptest", 12, [first, second], calib_dir=calibration
    )
    opened = source.open()

    assert opened.kind == "serial"
    assert opened.args == (str(first), str(second))
    assert psana.options == [("psana.calib-dir", str(calibration))]


def test_experiment_source_builds_slac_smd_dataset(monkeypatch):
    psana = FakePsana()
    monkeypatch.setitem(sys.modules, "psana", psana)
    source = Psana1RunSource.from_experiment("xpptest", 12, smd=True, mpi=True)

    opened = source.open()

    assert source.dataset == "exp=xpptest:run=12:smd"
    assert opened.kind == "mpi"
    assert opened.args == (source.dataset,)


def test_explicit_file_source_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="XTC file does not exist"):
        Psana1RunSource.from_files("xpptest", 12, [tmp_path / "missing.xtc"])


def test_detector_calibration_uses_psana_accessor(tmp_path, monkeypatch):
    xtc = tmp_path / "run.xtc"
    xtc.touch()
    calibration = tmp_path / "calib"
    calibration.mkdir()
    psana = FakePsana()
    monkeypatch.setitem(sys.modules, "psana", psana)
    source = Psana1RunSource.from_files(
        "xpptest", 12, [xtc], calib_dir=calibration
    )

    values = detector_calibration(12, "pixel_rms", source=source)

    assert values.shape == (3, 2, 2, 2)
    assert np.all(values == np.float32(1.2))
    assert psana.options == [("psana.calib-dir", str(calibration))]
    assert psana.calls == [("serial", (str(xtc),))]


def test_detector_calibration_rejects_unknown_constant():
    with pytest.raises(ValueError, match="unsupported calibration constant"):
        detector_calibration(12, "pixel_gain")
