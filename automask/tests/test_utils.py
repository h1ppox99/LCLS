import os
import sys

import numpy as np

import automask.utils as utils


def test_configure_psana_environment_adds_smalldata_to_running_kernel(
    monkeypatch, tmp_path
):
    checkout = tmp_path / "smalldata_tools_checkout"
    (checkout / "smalldata_tools").mkdir(parents=True)
    monkeypatch.setenv("SIT_PSDM_DATA", str(tmp_path / "psdm"))
    monkeypatch.setenv("SMALLDATA_TOOLS", str(checkout))
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    try:
        environment = utils.configure_psana_environment()

        assert environment["SMALLDATA_TOOLS"] == str(checkout)
        assert sys.path[0] == str(checkout)
        assert os.environ["PYTHONPATH"].split(os.pathsep) == [
            str(checkout),
            "/existing/path",
        ]
    finally:
        if str(checkout) in sys.path:
            sys.path.remove(str(checkout))


class FakeSource:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeEventKey:
    def __init__(self, payload_type, source, alias=""):
        self._payload_type = payload_type
        self._source = FakeSource(source)
        self._alias = alias

    def type(self):
        return self._payload_type

    def src(self):
        return self._source

    def alias(self):
        return self._alias

    def key(self):
        return ""


class FakeAnalog:
    def __init__(self, values):
        self.values = values

    def channelVoltages(self):
        return self.values


class FakeFifo:
    def __init__(self, code):
        self.code = code

    def eventCode(self):
        return self.code


class FakeEvr:
    def __init__(self, codes):
        self.codes = codes

    def fifoEvents(self):
        return [FakeFifo(code) for code in self.codes]


AnalogType = type("BldDataAnalogInputV1", (), {"__module__": "psana.Bld"})
EvrType = type("DataV4", (), {"__module__": "psana.EvrData"})


class FakeEvent:
    def __init__(self, analog, codes):
        self.analog_key = FakeEventKey(AnalogType, "BldInfo(TEST-AIN)")
        self.evr_key = FakeEventKey(
            EvrType, "DetInfo(NoDetector.0:Evr.0)", alias="evr0"
        )
        self.payloads = {
            AnalogType: FakeAnalog(analog),
            EvrType: FakeEvr(codes),
        }

    def keys(self):
        return [self.analog_key, self.evr_key]

    def get(self, payload_type, _source):
        return self.payloads[payload_type]


class FakeEpicsStore:
    def __init__(self, states):
        self.states = states
        self.index = 0

    def pvNames(self):
        return ["XPP:TEST:DELAY.RBV", "XPP:TEST:STATE"]

    def alias(self, pv_name):
        return {
            "XPP:TEST:DELAY.RBV": "delay",
            "XPP:TEST:STATE": "state",
        }[pv_name]

    def value(self, alias):
        return self.states[self.index][alias]


class FakeEnvironment:
    def __init__(self, store):
        self.store = store

    def epicsStore(self):
        return self.store


class FakeDataSource:
    def __init__(self, events, epics_states):
        self._events = events
        self.store = FakeEpicsStore(epics_states)

    def events(self):
        for index, event in enumerate(self._events):
            self.store.index = index
            yield event

    def env(self):
        return FakeEnvironment(self.store)


class FakeRunSource:
    def __init__(self, data_source):
        self.data_source = data_source

    def open(self):
        return self.data_source


class FakeDetectorSet:
    covered_sources = {"TEST-AIN"}
    epics_metadata = {
        "delay": "XPP:TEST:DELAY.RBV",
        "state": "XPP:TEST:STATE",
    }

    def __init__(self, data_source):
        self.data_source = data_source

    def read(self, event):
        analog = event.payloads[AnalogType].values
        epics = self.data_source.store.states[self.data_source.store.index]
        return {
            "ai": {f"ch{index:02d}": value for index, value in enumerate(analog)},
            "epics": epics,
        }


def test_profile_run_values_discovers_and_profiles_payloads():
    events = [
        FakeEvent([0.0, 5.0], [90, 137]),
        FakeEvent([0.0, 0.0], [91, 137]),
    ]
    epics_states = [
        {"delay": 3.1, "state": "MOVING"},
        {"delay": 4.2, "state": "READY"},
    ]
    data_source = FakeDataSource(events, epics_states)
    source = FakeRunSource(data_source)
    profile = utils.profile_run_values(
        12, source=source, detector_set=FakeDetectorSet(data_source)
    )

    assert profile.events == 2
    analog = profile.values["ai/ch01"]
    beam = profile.values["DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"]
    code_90 = profile.values["DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[90]"]
    delay = profile.values["EPICS/delay"]
    state = profile.values["EPICS/state"]

    np.testing.assert_array_equal(analog, [5.0, 0.0])
    np.testing.assert_array_equal(beam, [1.0, 1.0])
    np.testing.assert_array_equal(code_90, [1.0, 0.0])
    np.testing.assert_array_equal(delay, [3.1, 4.2])
    np.testing.assert_array_equal(state, ["MOVING", "READY"])
    assert profile.epics[0]["PV"] == "XPP:TEST:DELAY.RBV"


def test_profile_run_values_can_stop_early():
    events = [FakeEvent([0.0, 5.0], [137]) for _ in range(3)]
    states = [{"delay": 1.0, "state": "READY"} for _ in events]
    data_source = FakeDataSource(events, states)

    profile = utils.profile_run_values(
        12,
        source=FakeRunSource(data_source),
        detector_set=FakeDetectorSet(data_source),
        max_events=2,
    )

    assert profile.events == 2
    assert all(values.shape == (2,) for values in profile.values.values())


def test_profile_summary_retains_constant_rows():
    events = [
        FakeEvent([0.0, 5.0], [137]),
        FakeEvent([0.0, 0.0], [137]),
    ]
    states = [
        {"delay": 1.0, "state": "READY"},
        {"delay": 2.0, "state": "READY"},
    ]
    data_source = FakeDataSource(events, states)

    profile = utils.profile_run_values(
        12,
        source=FakeRunSource(data_source),
        detector_set=FakeDetectorSet(data_source),
    )

    assert any(row["constant"] for row in profile.summary["xtc"])
    assert any(row["constant"] for row in profile.summary["epics"])
    assert all("summary" in row and "observed" not in row for row in profile.epics)


def test_profile_summary_reports_discrete_changes():
    coverage, summary, constant = utils._column_summary(
        np.asarray([0.0, 0.0, 1.0, 1.0, np.nan])
    )

    assert coverage == "80.0%"
    assert summary == "binary; 1 changes; 0 x 2, 1 x 2"
    assert constant is False


def test_short_values_rejects_opaque_and_binary_scalars():
    for value in (object(), b"header\x00payload", b"\xff"):
        with np.testing.assert_raises(ValueError):
            utils._short_values("field", value)

    assert utils._short_values("field", b"printable") == {"field": "printable"}


def test_payload_discovery_skips_unsupported_values_without_read_errors():
    class Payload:
        def useful(self):
            return 3

        def opaque(self):
            return object()

        def needs_argument(self, index):
            return index

        def broken(self):
            raise ValueError("cannot read")

        def filenames(self):
            raise AssertionError("psana bookkeeping accessor should be vetoed")

        def getFileNames(self):
            raise AssertionError("psana bookkeeping accessor should be vetoed")

    values, errors = utils._discover_payload_values(Payload())

    assert values == {"useful": 3}
    assert errors == 1


def test_profile_summary_bounds_high_cardinality_text():
    values = np.asarray([f"state-{index}" for index in range(20)], dtype=object)

    coverage, summary, constant = utils._column_summary(values)

    assert coverage == "100.0%"
    assert summary == "20 values; 19 changes; first=state-0, last=state-19"
    assert constant is False
