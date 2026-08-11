"""
features/catalog.py -- the default named features the masking stats depend on.

Each entry binds a catalogue name (what a stat's ``needs`` refers to) to a
reduction over a ShotSelection. This is the single place the implicit
"umean == mean over beam-on shots" convention -- previously scattered across
build_features, load_sample and the Sample docstring -- is written down once.

This catalogue is also where the experiment-specific selection policy lives.

Importing this module populates ``FEATURES``.
"""
from __future__ import annotations

from automask.features.base import FeatureSpec, register
from automask.shot_selection import Condition, PercentileTrim, ShotSelection

BEAM_ON_FIELD = "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"
SAMPLE_DIODE_FIELD = "diodeU/channels[0]"

# Experiment defaults live in the catalogue rather than in the generic selector.
LIT_SELECTION = ShotSelection(
    where=(Condition(BEAM_ON_FIELD, "==", 1),),
    trim=PercentileTrim(SAMPLE_DIODE_FIELD, low=0.03, high=0.03),
    n_shots=800,
)
DARK_SELECTION = ShotSelection(
    where=(Condition(BEAM_ON_FIELD, "==", 0),),
)

register(FeatureSpec("umean", "mean", LIT_SELECTION))
register(FeatureSpec("ustd", "std", LIT_SELECTION))
register(FeatureSpec("umad", "mad", LIT_SELECTION))
register(FeatureSpec("mean", "mean", DARK_SELECTION))
register(FeatureSpec("pedestal", source="calib", constant="pedestals",
                     gain=0, form="panel"))
register(FeatureSpec("pixel_rms", source="calib", constant="pixel_rms",
                     gain=0, form="panel"))
