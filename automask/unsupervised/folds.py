"""
unsupervised/folds.py -- per-pixel moments accumulated in K disjoint shot folds.

The reproducibility metrics need to ask "would this mask come out the same from
a DIFFERENT set of shots?". Answering that from the FeatureStore is impossible:
a feature there is keyed by a ShotSelection, and the selection has no notion of a
sub-sample, so every query returns the same 800 shots. Rather than widen
ShotSelection (which would rekey and invalidate every cached feature), this
module makes one XTC pass that accumulates the same per-pixel moments the store
does -- n, sum, sum-of-squares -- but keeps them SPLIT over K folds instead of
pooled. Everything downstream (split-half, bootstrap over folds, the event-axis
chi2) is then pure numpy on an 80 MB cache.

Folds are CONTIGUOUS blocks of the run in stream order, which makes two
different resamplings available from one accumulator:

    halves      folds {0..K/2-1} vs {K/2..K-1}   -- early run vs late run, so it
                                                    also catches drift, not just
                                                    sampling noise
    alternating even folds vs odd folds          -- interleaved in time, so drift
                                                    is shared and what is left is
                                                    close to pure sampling noise

The gap between those two is itself informative: if a mask reproduces across
alternating folds but not across halves, the detector state moved during the run
and no amount of averaging fixes it.

`fold_sample` rebuilds a full :class:`~automask.evaluation.Sample` from a chosen
subset of folds, so an unmodified Pipeline can be re-run on it.

Run:  python -m automask.unsupervised.folds 475      # build/refresh the cache
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from automask.shot_selection import ShotSelection

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../automask
CACHE_DIR = os.path.join(PKG, "outputs", "cache", "folds")
PANEL_SHAPE = (2, 512, 1024)
N_FOLDS = 10

# The lit-beam selection the production features use (features/catalog.py `_LIT`).
# Kept identical on purpose: the folds must resample the SAME population the
# pipeline is normally fed, or the stability they measure is not the pipeline's.
LIT = ShotSelection(beam="on")


@dataclass
class FoldMoments:
    """Per-pixel (count, sum, sum of squares) for each of K shot folds.

    Arrays are in native panel space `(K, 2, 512, 1024)`; `n` is `(K,)`. Moments
    are additive, so any subset of folds gives that subset's mean and variance
    exactly -- as if the run had been reduced over those shots alone.
    """
    run: int
    n: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    indices: np.ndarray          # event indices used, stream order
    fold_of_shot: np.ndarray     # fold id per entry of `indices`

    @property
    def k(self) -> int:
        return int(self.n.size)

    def moments(self, folds: Sequence[int]):
        """Pooled `(n, mean, std)` over `folds`, in panel space."""
        f = np.asarray(list(folds), dtype=int)
        n = float(self.n[f].sum())
        if n < 2:
            raise ValueError(f"folds {list(folds)} hold {n} shots; need >= 2")
        mean = self.s1[f].sum(axis=0) / n
        var = np.maximum(self.s2[f].sum(axis=0) / n - mean * mean, 0.0)
        return n, mean, np.sqrt(var)

    def halves(self):
        h = self.k // 2
        return list(range(h)), list(range(h, self.k))

    def alternating(self):
        return list(range(0, self.k, 2)), list(range(1, self.k, 2))

    def fold_means(self):
        """Per-fold per-pixel mean and its own standard error, `(K, ...)` each.

        The event-axis chi2 compares the scatter of these K numbers against the
        standard errors, which is a test of whether one pixel's shot-to-shot
        behaviour is consistent with a single stationary distribution.
        """
        n = self.n[:, None, None, None]
        mean = self.s1 / n
        var = np.maximum(self.s2 / n - mean * mean, 0.0)
        return mean, np.sqrt(var / np.maximum(n - 1, 1))


# ==========================================================================
#  build / cache
# ==========================================================================
def cache_path(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> str:
    from automask.features.base import FeatureSpec
    key = FeatureSpec("_", "mean", selection).content_key
    return os.path.join(CACHE_DIR, f"folds_{key}_k{k}_run{run:04d}.npz")


def build(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> FoldMoments:
    """One XTC pass -> fold moments, cached to `cache_path`."""
    from automask.io.read_xtc import iter_calibrated, scan_shots

    meta = scan_shots(run)
    indices = selection.resolve(meta)
    if indices.size < 2 * k:
        raise RuntimeError(f"run {run}: {indices.size} shots is too few for {k} folds")
    # Contiguous blocks in stream order (see module docstring).
    fold_of_shot = (np.arange(indices.size) * k // indices.size).astype(np.int32)
    by_event = dict(zip(indices.tolist(), fold_of_shot.tolist()))

    n = np.zeros(k, dtype=np.float64)
    s1 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    s2 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    used = 0
    for event_index, panel in iter_calibrated(run, indices):
        f = by_event[event_index]
        frame = panel.astype(np.float64)
        n[f] += 1
        s1[f] += frame
        s2[f] += frame * frame
        used += 1
        if used % 100 == 0:
            print(f"[folds] run {run:04d}: {used}/{indices.size} frames", flush=True)
    if (n < 2).any():
        raise RuntimeError(f"run {run}: fold counts {n} -- a fold got < 2 frames")

    fm = FoldMoments(run=run, n=n, s1=s1, s2=s2, indices=indices,
                     fold_of_shot=fold_of_shot)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(run, k, selection)
    np.savez(path, n=n, s1=s1, s2=s2, indices=indices, fold_of_shot=fold_of_shot,
             run=run)
    print(f"[folds] run {run:04d}: {used} frames over {k} folds -> {path}")
    return fm


def load(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> FoldMoments:
    """Cached fold moments for `run`, built from XTC on a miss."""
    path = cache_path(run, k, selection)
    if not os.path.exists(path):
        return build(run, k, selection)
    z = np.load(path)
    return FoldMoments(run=run, n=z["n"], s1=z["s1"], s2=z["s2"],
                       indices=z["indices"], fold_of_shot=z["fold_of_shot"])


# ==========================================================================
#  a Sample restricted to a subset of shots
# ==========================================================================
def with_shot_features(base, mean_asm: np.ndarray, std_asm: np.ndarray, scale: float):
    """Copy of Sample `base` with its shot-derived fields replaced.

    Replaced: `umean`, `ustd` (the reductions the store serves) and `sumimg`
    (the run-sum image every intensity stat reads, rebuilt as `mean * scale`).
    Left alone: `human`, `calib`, `pedestal`, `pixel_rms` -- none of those is
    estimated from this run's shots, so resampling shots must not perturb them.

    `sumimg`'s zero set is forced to match the frozen sum image: `real` is a
    geometry fact, and letting it flicker with the shot sample would show up as
    fake instability in the geometry floor, which is not what any of these
    metrics is trying to measure.
    """
    sumimg = np.asarray(mean_asm, dtype=np.float64) * float(scale)
    sumimg[base.sumimg == 0] = 0.0
    return dataclasses.replace(
        base, sumimg=sumimg,
        umean=np.asarray(mean_asm, dtype=np.float64),
        ustd=np.asarray(std_asm, dtype=np.float64))


def fold_sample(base, fm: FoldMoments, folds: Sequence[int]):
    """Sample `base` rebuilt from the shots in `folds` alone.

    The `sumimg` scale is the FIXED total shot count rather than the subset's
    own, so the two sides of a split carry the same overall magnitude: every
    stat reading it is a robust z and so scale-free anyway, but pinning the scale
    means a difference between halves can only come from the pixels.
    """
    from automask.geometry import panel_to_asm

    n, mean, std = fm.moments(folds)
    return with_shot_features(base, panel_to_asm(mean, fm.run),
                              panel_to_asm(std, fm.run), float(fm.n.sum()))


def main(runs: Optional[Sequence[int]] = None, k: int = N_FOLDS):
    from automask.evaluation import EVAL_RUNS
    for run in (EVAL_RUNS if runs is None else runs):
        fm = load(run, k)
        n, mean, std = fm.moments(range(fm.k))
        print(f"  run {run}: {fm.k} folds, counts {fm.n.astype(int).tolist()}, "
              f"pooled mean {mean.mean():.3f}, pooled std {std.mean():.3f}")


if __name__ == "__main__":
    import sys
    main([int(a) for a in sys.argv[1:]] or None)
