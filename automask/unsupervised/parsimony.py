"""
unsupervised/parsimony.py -- tier 0: what a mask must satisfy to be arithmetic.

The weakest hypotheses in the package, and the cheapest. None of them looks at
the physics; they encode three facts that hold before any measurement:

  1. a mask that hides most of the detector has destroyed the experiment, and one
     that hides nothing cannot have found the defects that are demonstrably there
     (`mask_frac`, scored against a plausible band rather than as "less is more");
  2. the geometry+calib floor is 100%-precision by construction, so any mask
     failing to contain it is wrong on pixels we are certain about
     (`floor_containment`);
  3. detector defects are SPATIALLY COHERENT -- dead ASICs, wire shadows, hot
     clusters are blobs and lines, never salt-and-pepper -- so the pixels a mask
     adds beyond the floor should live in connected components, not scatter
     (`compactness`).

(3) is the only one of the three with real discriminating power, and it is worth
stating why it is not circular: it is a prior on defect MORPHOLOGY, independent
of the intensity evidence every detector in the pipeline actually thresholds. A
detector firing on noise produces isolated pixels, and this sees that.
"""
from __future__ import annotations

import numpy as np

from automask.unsupervised.base import MetricSpec, register_metric

MIN_BLOB = 4          # components smaller than this are speckle, not structure
PLAUSIBLE = (0.02, 0.25)   # fraction of the canvas a sane mask covers


def _added(cand, ctx):
    """The pixels this candidate masks BEYOND the floor -- what it is actually
    claiming, as opposed to what geometry and calibration already knew."""
    floor = ctx.floor()
    return cand.mask(ctx) & ~floor, floor


def mask_frac(cand, ctx) -> float:
    return float(cand.mask(ctx).mean())


def plausible_frac(cand, ctx) -> float:
    """1 inside the plausible band, decaying log-linearly outside it.

    `mask_frac` on its own has no direction -- both extremes are failures -- so
    the registered metric is this folded version, which is what a score card can
    actually rank on.
    """
    f = float(cand.mask(ctx).mean())
    lo, hi = PLAUSIBLE
    if f <= 0:
        return 0.0
    if lo <= f <= hi:
        return 1.0
    ratio = f / lo if f < lo else hi / f
    return float(max(0.0, 1.0 + np.log10(ratio)))


def floor_containment(cand, ctx) -> float:
    """Fraction of the 100%-precision floor the candidate keeps masked."""
    _, floor = _added(cand, ctx)
    n = int(floor.sum())
    return float((cand.mask(ctx) & floor).sum()) / n if n else 1.0


def compactness(cand, ctx, min_blob: int = MIN_BLOB) -> float:
    """Fraction of the added pixels living in components of >= `min_blob` px.

    Empty addition scores 1.0 (nothing claimed, nothing incoherent), which is
    correct for this metric alone and is exactly why tier 0 cannot stand by
    itself -- see the sufficiency note in base.py.
    """
    from scipy import ndimage

    added, _ = _added(cand, ctx)
    n = int(added.sum())
    if n == 0:
        return 1.0
    lab, k = ndimage.label(added, structure=np.ones((3, 3), dtype=int))
    if k == 0:
        return 1.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes[sizes >= min_blob].sum()) / n


register_metric(MetricSpec(
    name="plausible_frac", compute=plausible_frac, higher_is_better=True, tier=0,
    doc="masked fraction folded into a plausible band (0.02-0.25 of the canvas)"))
register_metric(MetricSpec(
    name="floor_containment", compute=floor_containment, higher_is_better=True,
    tier=0, doc="fraction of the geometry+calib floor the mask contains"))
register_metric(MetricSpec(
    name="compactness", compute=compactness, higher_is_better=True, tier=0,
    doc="fraction of added pixels in connected components of >= 4 px"))
