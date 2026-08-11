"""
stats/mad_variance.py -- robust-dispersion statistic on the lit-beam MAD.

Same construction as `stats/variance.py` -- signed robust-MAD z-score of the
log10 per-pixel dispersion over the beam-on shots -- with two deliberate
differences:

  * the dispersion is the ``mad`` reduction (1.4826-scaled median absolute
    deviation over shots) instead of ``std``. Same ShotSelection, so the two
    differ only in the estimator: MAD ignores the tails, so a pixel that is
    quiet most of the time but spikes on a handful of shots gets a small ``mad``
    and a large ``std``.
  * ``mode="high"``, i.e. the flagged side is LARGE dispersion, not small.

Because the MAD is insensitive to a minority of outlying shots, a pixel needs to
be *persistently* unstable across the bulk of the run to score high here -- the
target is the flickering/unstable-pixel class, not the dead/shadowed class that
``variance`` (mode="low") picks up. The two statistics are therefore
complementary rather than redundant, and both remain monotone in their feature,
so the log only conditions `k` and the TV stage (see the module docstring notes
in `stats/variance.py`).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from automask.stats.base import StatSpec, register_stat, robust_z


@dataclass
class MadVarianceParams:
    k: float = 3.5          # robust-MAD threshold
    mode: str = "high"      # flag LARGE robust dispersion (unstable pixels)


def mad_variance_stat(mad):
    """Signed robust-MAD z-score of log10(per-pixel MAD).

    z >> 0 = anomalously large robust dispersion (persistently unstable pixels),
    z << 0 = low dispersion (dead/shadowed) -- thresholded mode="high"."""
    return robust_z(mad, mad > 0, transform=np.log10)


def compute(sample, params: MadVarianceParams | None = None):
    return mad_variance_stat(sample.mad)


register_stat(StatSpec(
    name="mad_variance",
    compute=compute,
    params=MadVarianceParams,
    kind="field",
    mode="high",
    needs=("mad",),
    doc="robust-MAD z of log10 per-pixel MAD; high z == persistently unstable",
))
