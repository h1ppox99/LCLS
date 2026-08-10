import numpy as np

import automask.utils as utils


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


AnalogType = type(
    "BldDataAnalogInputV1", (), {"__module__": "psana.Bld"}
)
EvrType = type("DataV4", (), {"__module__": "psana.EvrData"})


class FakeEvent:
    def __init__(self, analog, codes):
        self.analog_key = FakeEventKey(
            AnalogType, "BldInfo(TEST-AIN)"
        )
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


def test_profile_run_values_discovers_and_profiles_payloads(monkeypatch):
    events = [
        FakeEvent([0.0, 5.0], [90, 137]),
        FakeEvent([0.0, 0.0], [91, 137]),
    ]
    epics_states = [
        {"delay": 3.1, "state": "MOVING"},
        {"delay": 4.2, "state": "READY"},
    ]
    monkeypatch.setattr(
        utils,
        "open_local_run",
        lambda _run: (FakeDataSource(events, epics_states), []),
    )
    monkeypatch.setattr(utils, "_show_table", lambda *_args: None)

    profile = utils.profile_run_values(12)

    assert profile["events"] == 2
    analog = profile["values"][
        "BldInfo(TEST-AIN)/Bld.BldDataAnalogInputV1/channelVoltages[1]"
    ]
    beam = profile["values"][
        "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"
    ]
    code_90 = profile["values"][
        "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[90]"
    ]
    delay = profile["values"]["EPICS/delay"]
    state = profile["values"]["EPICS/state"]

    np.testing.assert_array_equal(analog, [5.0, 0.0])
    np.testing.assert_array_equal(beam, [1.0, 1.0])
    np.testing.assert_array_equal(code_90, [1.0, 0.0])
    np.testing.assert_array_equal(delay, [3.1, 4.2])
    np.testing.assert_array_equal(state, ["MOVING", "READY"])
    assert profile["epics"][0]["PV"] == "XPP:TEST:DELAY.RBV"


def test_profile_summary_reports_discrete_changes():
    coverage, variation, observed = utils._column_summary(
        np.asarray([0.0, 0.0, 1.0, 1.0, np.nan])
    )

    assert coverage == "80.0%"
    assert variation == "2 values; 1 changes"
    assert observed == "0: 2, 1: 2"
