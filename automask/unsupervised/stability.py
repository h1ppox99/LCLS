"""
unsupervised/stability.py -- tier 1: does the mask reproduce?

The hypothesis is estimation-theoretic and needs no physics: the mask is an
ESTIMATE of a fixed property of the detector (which pixels are broken), computed
from a finite random sample of shots with an arbitrary choice of knobs. If the
same recipe run on other shots, or with slightly different knobs, returns a
different mask, then at least one of the two masks is reporting sampling noise as
a defect. Reproducibility is thus a necessary condition for correctness, and one
we can measure without ever knowing what the right answer is.

Three perturbations, in increasing order of honesty and cost:

  `stab_noise`   analytic. Each shot-derived field is jittered by its own
                 sampling error (mean by sigma/sqrt(n), std by sigma/sqrt(2(n-1))),
                 independently per pixel. Costs no I/O -- but it assumes the
                 errors are independent across pixels, which common-mode noise
                 and per-ASIC gain drift make false. Treat it as the cheap
                 surrogate, and see `metric_validation` for how far it agrees
                 with the real thing.
  `stab_alt`     real resampling, even vs odd SHOTS. Shots are dealt to folds
                 one at a time (`folds.py`), so both sides carry the same mix of
                 experimental conditions and what is left is close to pure shot
                 sampling. This used to interleave 80-shot BLOCKS instead, which
                 balanced nothing: the CC/VCC branch clusters in runs of a
                 thousand-odd shots, so block parity left the two sides with
                 branch compositions differing by up to 0.15.
  `stab_time`    real resampling, contiguous halves: first half of the run vs
                 second. Everything `stab_alt` sees, PLUS any drift in the
                 detector state AND any change in condition mix between the two
                 halves. `stab_alt - stab_time` therefore isolates
                 non-stationarity in the broad sense, and a mask can only be a
                 run-level constant if that gap is small.

  `stab_hyper`   the knobs, not the data: every float hyperparameter in the
                 recipe is multiplied by lognormal(eps) and the mask recomputed.
                 A threshold sitting on a plateau barely moves; one perched on a
                 cliff -- which is what over-tuning to a particular run looks
                 like -- falls off it.

All four report a mean IoU in [0, 1], higher = more reproducible. All four are
PROCEDURE metrics (`needs_maker=True`): they re-run the recipe, so they say
nothing about a mask handed over as a bare array. And all four are maximized by
recipes that decide nothing at all -- mask everything, or mask only the floor --
so they rank candidates only in combination with a tier-2/3 quality term.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

from automask.unsupervised.base import MetricSpec, iou, register_metric
from automask.unsupervised.folds import fold_sample, with_shot_images

N_NOISE = 3           # analytic-noise repetitions
N_HYPER = 4           # hyperparameter-jitter repetitions
EPS_HYPER = 0.10      # relative scale of the knob jitter (lognormal sigma)

# Knobs that are facts about the hardware or the fusion contract rather than
# tuning choices, and so must not be jittered: the Jungfrau ASIC really is
# 256 px, and `defectiveness_scale` is a modelling decision documented in
# masking.py, not a fitted value.
FIXED_KNOBS = frozenset({"asic", "defectiveness_scale"})


# ==========================================================================
#  perturbing the DATA
# ==========================================================================
def n_shots(run: int, selection=None, default: int = 800) -> int:
    """Number of frames in a cached selected-shot standard deviation."""
    from automask.image_store import ImageStore
    from automask.selection_presets import BEAM_ON_SELECTION

    selection = selection or BEAM_ON_SELECTION
    counts = ImageStore().counts(run, selection, "std")
    if not counts:
        return default
    return int(counts.get("n_used") or counts.get("n_selected") or default)


def sum_scale(sample) -> float:
    """The ratio `sumimg / mean` -- how many shots' worth the frozen sum image
    is, in the units the mean is served in. Taken as a median over live pixels
    rather than assumed, because the sum and selected-shot images were frozen by
    different producers and need not share a shot count."""
    real = sample.real & (sample.mean != 0)
    if not real.any():
        raise ValueError("no pixels with both a sum and a nonzero mean")
    return float(np.median(sample.sumimg[real] / sample.mean[real]))


def noisy_sample(sample, rng, n: Optional[int] = None):
    """`sample` with its shot-derived fields redrawn at their sampling error.

    mean ~ N(mean, sigma^2/n) and std ~ std * (1 + N(0, 1)/sqrt(2(n-1))) are the
    standard large-n sampling laws for the first two moments of n draws. Both
    are applied per pixel independently -- the approximation this metric makes,
    and the reason the fold-based versions exist.
    """
    n = n or n_shots(int(sample.run), sample.selection)
    if sample.mean is None or sample.std is None:
        raise ValueError(
            "stab_noise needs the 'mean' and 'std' reductions; load_sample was "
            "called without them")
    sd = np.asarray(sample.std, dtype=np.float64)
    mean = np.asarray(sample.mean, dtype=np.float64)
    mean_p = mean + rng.standard_normal(mean.shape) * sd / np.sqrt(n)
    sd_p = sd * np.maximum(
        1.0 + rng.standard_normal(sd.shape) / np.sqrt(2.0 * (n - 1)), 0.0)
    return with_shot_images(sample, mean_p, sd_p, sum_scale(sample))


# ==========================================================================
#  perturbing the KNOBS
# ==========================================================================
def jitter_params(params, eps: float, rng):
    """Multiply every free numeric field of a params dataclass by lognormal(eps).

    Integers stay integers and stay >= 1 (a 0-px structuring element or a 0-vote
    Hough threshold is not a perturbation of the recipe, it is a different
    recipe). Strings, bools, None and `FIXED_KNOBS` are left alone.
    """
    if params is None or not dataclasses.is_dataclass(params):
        return params
    changes = {}
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        if f.name in FIXED_KNOBS or isinstance(v, (bool, str)) or v is None:
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            continue
        new = float(v) * float(np.exp(rng.normal(0.0, eps)))
        changes[f.name] = max(1, int(round(new))) if isinstance(v, (int, np.integer)) \
            else new
    return dataclasses.replace(params, **changes) if changes else params


def _jitter_slot(params, eps, rng):
    """Jitter a regularizer params slot, which may be a single object or a list
    aligned with a list of regularizer names (see masking.Detector._stages)."""
    if isinstance(params, (list, tuple)):
        return [jitter_params(p, eps, rng) for p in params]
    return jitter_params(params, eps, rng)


def jitter_pipeline(pipe, eps: float, rng):
    """A copy of `pipe` with every free knob -- stat, regularizers, combiner --
    independently perturbed. Structure (which detectors, which regularizers, the
    combiner) is untouched: this asks whether the recipe is on a plateau, not
    whether a different recipe would do better."""
    from automask.masking import Detector, Pipeline

    dets = [Detector(stat=d.stat,
                     stat_params=jitter_params(d._params(), eps, rng),
                     field_reg=d.field_reg,
                     field_reg_params=_jitter_slot(d.field_reg_params, eps, rng),
                     mask_reg=d.mask_reg,
                     mask_reg_params=_jitter_slot(d.mask_reg_params, eps, rng))
            for d in pipe.detectors]
    return Pipeline(detectors=dets, shot_selection=pipe.shot_selection,
                    floor_stats=list(pipe.floor_stats),
                    combiner=pipe.combiner,
                    combiner_params=jitter_params(pipe.combiner_params, eps, rng))


def pipeline_jitter(pipe) -> Callable:
    """The `Candidate.jitter` hook for a pipeline-backed candidate."""
    return lambda eps, rng: jitter_pipeline(pipe, eps, rng).run


# ==========================================================================
#  metrics
# ==========================================================================
def _require_procedure(cand):
    if cand.static:
        raise ValueError(
            f"candidate {cand.name!r} is a static mask: it ignores the shots, so "
            f"a reproducibility score would be 1.0 by construction")


def stab_noise(cand, ctx, reps: int = N_NOISE) -> float:
    """Mean IoU between the nominal mask and masks from analytically-jittered
    inputs."""
    _require_procedure(cand)
    base = cand.mask(ctx)
    rng = ctx.rng("noise")
    return float(np.mean([iou(base, cand.make(noisy_sample(ctx.sample, rng)))
                          for _ in range(reps)]))


def _split(cand, ctx, which: str) -> float:
    _require_procedure(cand)
    fm = ctx.folds()
    # `alt` reads the DEALT axis and `time` the chronological one; the two carry
    # the same shots partitioned oppositely, and reading the wrong axis returns a
    # perfectly plausible number for the other question.
    dealt = which == "alt"
    a, b = fm.alternating() if dealt else fm.halves()
    return iou(cand.make(fold_sample(ctx.sample, fm, a, dealt=dealt)),
               cand.make(fold_sample(ctx.sample, fm, b, dealt=dealt)))


def stab_alt(cand, ctx) -> float:
    """IoU between masks built from even- vs odd-numbered SHOTS.

    Shots are dealt to folds one at a time, so the two sides see the same mix of
    experimental conditions -- see `folds.py` for the measurement that forced
    this to be a deal over shots rather than an interleave over blocks."""
    return _split(cand, ctx, "alt")


def stab_time(cand, ctx) -> float:
    """IoU between masks built from the first vs the second half of the run."""
    return _split(cand, ctx, "time")


def stab_hyper(cand, ctx, reps: int = N_HYPER, eps: float = EPS_HYPER) -> float:
    """Mean IoU between the nominal mask and masks from jittered knobs."""
    _require_procedure(cand)
    if cand.jitter is None:
        raise ValueError(f"candidate {cand.name!r} carries no jitter recipe")
    base = cand.mask(ctx)
    rng = ctx.rng("hyper")
    return float(np.mean([iou(base, cand.jitter(eps, rng)(ctx.sample))
                          for _ in range(reps)]))


register_metric(MetricSpec(
    name="stab_noise", compute=stab_noise, higher_is_better=True, tier=1,
    needs_maker=True,
    doc="IoU under analytic per-pixel resampling of selected-shot images"))
register_metric(MetricSpec(
    name="stab_alt", compute=stab_alt, higher_is_better=True, tier=1,
    needs_maker=True, doc="IoU between masks from even vs odd shot blocks"))
register_metric(MetricSpec(
    name="stab_time", compute=stab_time, higher_is_better=True, tier=1,
    needs_maker=True, doc="IoU between masks from the first vs second half of the run"))
register_metric(MetricSpec(
    name="stab_hyper", compute=stab_hyper, higher_is_better=True, tier=1,
    needs_maker=True, doc="IoU under 10% lognormal jitter of every free knob"))
