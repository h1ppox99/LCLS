"""
features/catalog.py -- the default named features the masking stats depend on.

Each entry binds a catalogue name (what a stat's ``needs`` refers to) to a
reduction over a ShotSelection. This is the single place the implicit
"umean == mean over x-ray-on shots" convention -- previously scattered across
build_features, load_sample and the Sample docstring -- is written down once.

Importing this module populates ``FEATURES``.
"""
from __future__ import annotations

from automask.features.base import FeatureSpec, register
from automask.shot_selection import ShotSelection

# One selection reused for both lit reductions (mean AND std over the same shots).
_LIT = ShotSelection(xray="on")
_DARK = ShotSelection(xray="off")

register(FeatureSpec("umean", "mean", _LIT))   # lit-beam per-pixel mean
register(FeatureSpec("ustd", "std", _LIT))     # lit-beam per-pixel std
register(FeatureSpec("umad", "mad", _LIT))     # lit-beam per-pixel MAD (robust std)
register(FeatureSpec("mean", "mean", _DARK))   # beam-off dark frame
