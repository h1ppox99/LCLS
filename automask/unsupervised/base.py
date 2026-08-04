"""
unsupervised/base.py -- registry + shared context for label-free mask metrics.

In production there is no human mask. Everything in this package is therefore a
SURROGATE for `IoU vs human`, built only from the run's own data, and each one is
only worth what it can be shown to be worth -- `studies/metric_validation.py`
measures every metric here against the ground truth we do have (runs 389/475) and
reports its rank correlation with the true IoU. A metric that does not rank
variants like the truth does is not a metric, it is a number.

TWO KINDS OF METRIC, and the difference matters.

  * A MASK metric scores the array: `compute(candidate, ctx)` reads
    `candidate.mask(ctx)`. Physics consistency (azimuthal) and event-axis
    anomaly are of this kind, and they can rank any mask from any source.
  * A PROCEDURE metric scores the recipe that produced the mask: it re-runs
    `candidate.make` on perturbed inputs or perturbed knobs and asks whether the
    answer moves. It is undefined for a mask handed over as a bare array -- a
    frozen array is trivially "stable" -- which is why `Candidate` carries a
    maker rather than an array, and why `MetricSpec.needs_maker` exists.

Both live in one registry because the score card mixes them, but conflating them
is the easiest way to fool yourself: stability alone is maximized by masking
nothing (or everything), so it is a NECESSARY, not sufficient, condition and only
means something read together with a tier-2/3 quality term.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

METRICS: Dict[str, "MetricSpec"] = {}


@dataclass
class MetricSpec:
    """One label-free metric.

    `higher_is_better` is what lets the validation study compare metrics on one
    axis (it flips the sign before correlating with IoU). `tier` records how much
    the metric assumes: 0 = arithmetic sanity, 1 = the mask is an estimate and
    should reproduce, 2 = the scattering is azimuthally isotropic, 3 = a pixel's
    shot-to-shot behaviour is stationary.
    """
    name: str
    compute: Callable[["Candidate", "MetricContext"], float]
    higher_is_better: bool
    tier: int
    needs_maker: bool = False
    doc: str = ""


def register_metric(spec: MetricSpec) -> MetricSpec:
    METRICS[spec.name] = spec
    return spec


# ==========================================================================
#  candidate -- a mask, plus the recipe that made it
# ==========================================================================
@dataclass
class Candidate:
    """A mask under test, carried as the FUNCTION that produces it.

    `make(sample) -> bool array` is required: the reproducibility metrics need to
    re-run it on resampled shots. `jitter(eps, rng) -> make`-like callable is
    optional and returns the same recipe with its hyperparameters perturbed by
    relative scale `eps`; candidates without one simply score NaN on the
    hyperparameter metric instead of silently scoring perfectly.

    `static` marks a mask that does not depend on the run's shots at all (a
    frozen array, the human reference). Such a candidate would score a perfect
    1.0 on every tier-1 metric for the trivial reason that it ignores its input,
    so the procedure metrics refuse it and report NaN instead -- a vacuous
    perfect score is worse than a missing one.
    """
    name: str
    make: Callable
    jitter: Optional[Callable] = None
    static: bool = False
    note: str = ""
    _masks: dict = field(default_factory=dict, repr=False)

    def mask(self, ctx: "MetricContext") -> np.ndarray:
        """This candidate's mask on `ctx.sample`, computed once per run."""
        key = ctx.run
        if key not in self._masks:
            self._masks[key] = np.asarray(self.make(ctx.sample), dtype=bool)
        return self._masks[key]


# ==========================================================================
#  context -- the run, and the expensive things metrics share
# ==========================================================================
@dataclass
class MetricContext:
    """Per-run inputs shared by every metric, with the costly derived products
    computed at most once (the azimuthal per-pixel frame costs ~2 s, the fold
    moments an XTC pass on a cold cache)."""
    sample: object
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def run(self) -> int:
        return int(self.sample.run)

    def rng(self, tag: str = "") -> np.random.Generator:
        """A generator seeded from (seed, run, tag) -- reproducible, and
        independent across metrics so one metric's draws cannot shift another's."""
        return np.random.default_rng(abs(hash((self.seed, self.run, tag))) % (2**32))

    def cached(self, key: str, build: Callable):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def folds(self):
        from automask.unsupervised import folds as F
        return self.cached("folds", lambda: F.load(self.run))

    def floor(self) -> np.ndarray:
        """The production geometry+calib floor: the reference partition every
        control draw and every ring/sector boundary is defined against."""
        from automask.masking import production_pipeline
        return self.cached("floor",
                           lambda: production_pipeline().floor(self.sample))

    def azimuthal_frame(self):
        from automask.unsupervised.azimuthal import build_frame
        return self.cached("azframe",
                           lambda: build_frame(self.sample, self.floor()))


# ==========================================================================
#  scoring
# ==========================================================================
def score_candidate(cand: Candidate, ctx: MetricContext,
                    metrics: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Every registered metric on one candidate. A metric that raises records
    NaN and the reason rather than aborting the sweep -- an unusable variant
    (empty mask, no jitter recipe) is a legitimate outcome to report."""
    names = list(METRICS) if metrics is None else list(metrics)
    out: Dict[str, float] = {}
    for name in names:
        spec = METRICS[name]
        try:
            out[name] = float(spec.compute(cand, ctx))
        except Exception as e:                    # noqa: BLE001 -- reported, not hidden
            out[name] = float("nan")
            out[f"{name}__error"] = f"{type(e).__name__}: {e}"
    return out


# ==========================================================================
#  shared numerics
# ==========================================================================
def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two boolean masks; 1.0 when both are empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0
