"""Official SLAC detector adapters for psana1 event profiling."""

from __future__ import annotations


class Lcls1DetectorAdapters:
    """XPP detector adapters backed by ``smalldata_tools``."""

    def __init__(self, data_source):
        try:
            from smalldata_tools.lcls1.default_detectors import (
                detData,
                ebeamDetector,
                epicsDetector,
                gasDetector,
            )
            from smalldata_tools.lcls1.hutch_default import defaultDetectors
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "smalldata_tools is required for run profiling. Clone "
                "https://github.com/slac-lcls/smalldata_tools and set "
                "SMALLDATA_TOOLS in psana_env.local."
            ) from error

        detectors = defaultDetectors("xpp", env=data_source.env())
        detectors = [
            detector
            for detector in detectors
            if getattr(detector, "name", None) not in {"lightStatus", "epics", "l3t"}
        ]
        detectors.extend((ebeamDetector(), gasDetector()))

        epics_store = data_source.env().epicsStore()
        epics_pvs = []
        self.epics_metadata = {}
        for pv_name in epics_store.pvNames():
            alias = epics_store.alias(pv_name) or pv_name
            epics_pvs.append((pv_name, alias))
            self.epics_metadata[alias] = pv_name
        detectors.append(epicsDetector(name="epics", PVlist=epics_pvs))

        self.detectors = detectors
        self.covered_sources = {
            detector.detname
            for detector in detectors
            if getattr(detector, "detname", None) not in {None, "epics"}
        }
        self._det_data = detData

    def read(self, event):
        return self._det_data(self.detectors, event)
