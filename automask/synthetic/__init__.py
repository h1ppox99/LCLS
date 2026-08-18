"""
automask.synthetic -- synthetic-artifact evaluation for the masking code.

A small, deliberately minimal harness: inject one deterministic synthetic
artifact into the *valid* pixels of a real diffraction image, run a masking
function on the corrupted image, and measure whether the injected artifact is
recovered -- scored only over pixels that were valid in the ground truth.

  * artifacts.py -- the artifact generators (+ the ARTIFACTS registry)
  * metrics.py   -- region-restricted precision/recall/F1/IoU/FPR/masked-frac
  * evaluate.py  -- config -> per-example CSV + figures + aggregate summary
                    (python -m automask.synthetic.evaluate)

Mask convention throughout matches the rest of automask: bool, True == masked.
"""

from automask.synthetic.artifacts import (
    ARTIFACTS,
    straight_streak,
    beamstop_shadow,
)
from automask.synthetic.metrics import masking_metrics

__all__ = [
    "ARTIFACTS",
    "straight_streak",
    "beamstop_shadow",
    "masking_metrics",
]
