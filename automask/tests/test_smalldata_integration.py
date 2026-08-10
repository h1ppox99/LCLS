import sys
from types import ModuleType, SimpleNamespace

from automask.io.smalldata import Lcls1SmallDataDetectors


class Adapter:
    def __init__(self, name, detname):
        self.name = name
        self.detname = detname


class EpicsStore:
    def pvNames(self):
        return ["XPP:TEST:ONE", "XPP:TEST:TWO"]

    def alias(self, pv):
        return "one" if pv.endswith("ONE") else ""


def test_lcls1_detector_set_reuses_smalldata_adapters(monkeypatch):
    package = ModuleType("smalldata_tools")
    lcls1 = ModuleType("smalldata_tools.lcls1")
    defaults = ModuleType("smalldata_tools.lcls1.default_detectors")
    hutch = ModuleType("smalldata_tools.lcls1.hutch_default")

    defaults.detData = lambda detectors, event: {
        detector.name: {"value": event} for detector in detectors
    }
    defaults.ebeamDetector = lambda: Adapter("ebeam", "EBeam")
    defaults.gasDetector = lambda: Adapter("gas_detector", "FEEGasDetEnergy")

    def epics_detector(name, PVlist):
        detector = Adapter(name, "epics")
        detector.pvs = PVlist
        return detector

    defaults.epicsDetector = epics_detector
    hutch.defaultDetectors = lambda _hutch, env: [
        Adapter("lightStatus", "NoDetector.0:Evr.0"),
        Adapter("epics", "epics"),
        Adapter("ai", "XPP-AIN-01"),
    ]
    monkeypatch.setitem(sys.modules, "smalldata_tools", package)
    monkeypatch.setitem(sys.modules, "smalldata_tools.lcls1", lcls1)
    monkeypatch.setitem(
        sys.modules, "smalldata_tools.lcls1.default_detectors", defaults
    )
    monkeypatch.setitem(sys.modules, "smalldata_tools.lcls1.hutch_default", hutch)

    environment = SimpleNamespace(epicsStore=lambda: EpicsStore())
    data_source = SimpleNamespace(env=lambda: environment)
    detectors = Lcls1SmallDataDetectors(data_source)

    assert [detector.name for detector in detectors.detectors] == [
        "ai",
        "ebeam",
        "gas_detector",
        "epics",
    ]
    assert detectors.covered_sources == {"XPP-AIN-01", "EBeam", "FEEGasDetEnergy"}
    assert detectors.epics_metadata == {
        "one": "XPP:TEST:ONE",
        "XPP:TEST:TWO": "XPP:TEST:TWO",
    }
    assert detectors.read(3)["ai"] == {"value": 3}
