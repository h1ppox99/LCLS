"""Experiment-specific shot selections for xppl1016922."""

from __future__ import annotations

from automask.selection.shot_selection import Condition, PercentileTrim, ShotSelection

BEAM_ON_FIELD = "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"
SAMPLE_DIODE_FIELD = "diodeU/channels[0]"

BEAM_ON_SELECTION = ShotSelection(
    where=(Condition(BEAM_ON_FIELD, "==", 1),),
    trim=PercentileTrim(SAMPLE_DIODE_FIELD, low=0.03, high=0.03),
    n_shots=800,
)

BEAM_OFF_SELECTION = ShotSelection(
    where=(Condition(BEAM_ON_FIELD, "==", 0),),
)
