"""
unsupervised/folds.py -- per-pixel moments accumulated in K disjoint shot folds.

The reproducibility metrics need to ask "would this mask come out the same from
a DIFFERENT set of shots?". An ImageStore reduction is keyed by a ShotSelection,
and the selection has no notion of a sub-sample, so every query returns the same
800 shots. Rather than widen ShotSelection (which would rekey cached images), this
module makes one XTC pass that accumulates the same per-pixel moments the store
does -- n, sum, sum-of-squares -- but keeps them SPLIT over K folds instead of
pooled. Everything downstream (split-half, bootstrap over folds, the event-axis
chi2) is then pure numpy on an 80 MB cache.

TWO fold axes are accumulated, not one, because the two questions this cache is
asked need opposite things from a fold.

    TIME BLOCKS  contiguous runs of shots in stream order. `halves()` (early vs
                 late) and the tier-3 cross-fold chi2 both need chronology: a
                 stationarity test over folds that each span the whole run is
                 not a stationarity test.
    DEALT FOLDS  shots handed out round-robin, one to each fold in turn.
                 `alternating()` needs the opposite -- folds that are
                 INTERCHANGEABLE, so that a difference between two of them is
                 sampling noise and nothing else.

The gap between the two splits is the informative quantity: if a mask reproduces
across dealt folds but not across halves, the run moved and no amount of
averaging fixes it.

WHY DEALING, AND WHY IT IS NOT THE SAME AS INTERLEAVING BLOCKS. This module used
to build contiguous blocks and then interleave whole blocks (even folds vs odd
folds), calling that the drift-balanced split. It is not. The CC/VCC branch --
the dominant per-shot condition in this experiment -- does not vary shot to shot;
it clusters, with runs of 1000-1700 shots in acquisition order (measured in
`studies/loss_identification.py::exp_a2_interchangeable`, runs 378/389/396).
Against blocks of that length, interleaving 80-shot blocks balances nothing in
particular: over ten folds the VCC-open fraction spanned 0.425, 0.597 and 0.175
on the three runs, and on two of them the block-alternating split came out WORSE
balanced than the contiguous-halves split it existed to improve on.

Dealing shot by shot fixes it, and the reason it works is exactly the assumption
being relied on: adjacent shots share a condition 98.6-99.7% of the time against
~80% for two shots drawn at random, so handing consecutive shots to different
folds gives every fold the same mix. Dealt, the same spread falls to 0.025,
0.016 and 0.050. Note `(i mod K)` even/odd is shot parity for even K, so
`alternating()` over dealt folds is the ABAB split.

The one way dealing fails is a condition whose period is COMMENSURATE with K --
a branch flipping every ten shots would put fold 0 on one branch and fold 5 on
the other. Here the block length is 7-34x the fold count, so every fold samples
every block; that ratio is worth re-checking on a run whose branch pattern looks
different.

Cost: two fold axes means two accumulators, so the cache is 2x the size (~340 MB
per run) and one extra add per frame. The XTC pass, which is what actually
costs, is still one.

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

from automask.image_store import _content_key
from automask.selection_presets import BEAM_ON_SELECTION
from automask.shot_selection import ShotSelection

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../automask
CACHE_DIR = os.path.join(PKG, "outputs", "cache", "folds")
PANEL_SHAPE = (2, 512, 1024)
N_FOLDS = 10

LIT = BEAM_ON_SELECTION


@dataclass
class FoldMoments:
    """Per-pixel (count, sum, sum of squares) on TWO fold axes.

    `n/s1/s2` are the chronological time blocks and `dn/ds1/ds2` the dealt
    folds; arrays are native panel space `(K, 2, 512, 1024)` with `(K,)` counts.
    Moments are additive, so any subset of either axis gives that subset's mean
    and variance exactly -- as if the run had been reduced over those shots
    alone. The two axes partition the same shots, so they sum to the same total.
    """
    run: int
    n: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    indices: np.ndarray          # event indices used, stream order
    block_of_shot: np.ndarray    # time-block id per entry of `indices`
    dn: np.ndarray               # dealt-fold counts
    ds1: np.ndarray
    ds2: np.ndarray
    fold_of_shot: np.ndarray     # dealt-fold id per entry of `indices`

    @property
    def k(self) -> int:
        return int(self.n.size)

    def moments(self, folds: Sequence[int], dealt: bool = False):
        """Pooled `(n, mean, std)` over `folds` on the chosen axis.

        `dealt=False` reads the chronological blocks, `dealt=True` the dealt
        folds. The flag is explicit rather than inferred because the two axes
        have identical shapes and answer opposite questions -- silently reading
        the wrong one gives a plausible number.
        """
        n_arr, s1, s2 = ((self.dn, self.ds1, self.ds2) if dealt
                         else (self.n, self.s1, self.s2))
        f = np.asarray(list(folds), dtype=int)
        n = float(n_arr[f].sum())
        if n < 2:
            raise ValueError(f"folds {list(folds)} hold {n} shots; need >= 2")
        mean = s1[f].sum(axis=0) / n
        var = np.maximum(s2[f].sum(axis=0) / n - mean * mean, 0.0)
        return n, mean, np.sqrt(var)

    def halves(self):
        """Early run vs late run, as TIME-BLOCK indices."""
        h = self.k // 2
        return list(range(h)), list(range(h, self.k))

    def alternating(self):
        """Shot parity, as DEALT-fold indices -- pass with `dealt=True`.

        For even K, `(i mod K)` even is exactly `i` even, so this is the ABAB
        split over shots rather than over blocks.
        """
        return list(range(0, self.k, 2)), list(range(1, self.k, 2))

    def fold_means(self):
        """Per-TIME-BLOCK per-pixel mean and its own standard error.

        Chronological on purpose: the event-axis chi2 compares the scatter of
        these K numbers against their standard errors to ask whether a pixel is
        stationary over the run. Dealt folds each span the whole run, so the
        same chi2 computed over them would test overdispersion, not stationarity
        -- a different hypothesis wearing the same arithmetic.
        """
        n = self.n[:, None, None, None]
        mean = self.s1 / n
        var = np.maximum(self.s2 / n - mean * mean, 0.0)
        return mean, np.sqrt(var / np.maximum(n - 1, 1))


# ==========================================================================
#  build / cache
# ==========================================================================
#: Bumped when the cache LAYOUT changes, so a stale file is rebuilt instead of
#: being read with the wrong meaning. v2 added the dealt fold axis.
LAYOUT = "v2"


def cache_path(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> str:
    key = _content_key(selection, "mean")
    return os.path.join(CACHE_DIR, f"folds_{LAYOUT}_{key}_k{k}_run{run:04d}.npz")


def assign_folds(n_shots: int, k: int):
    """`(time block, dealt fold)` id for each shot, in stream order.

    Time blocks are contiguous and stay local to this module -- they answer a
    stationarity question, not a resampling one. The dealt axis defers to
    `ShotSelection.create_k_folds` so there is one definition of the deal.
    See the module docstring for why both are kept.
    """
    i = np.arange(n_shots)
    fold = np.empty(n_shots, dtype=np.int32)
    for f, pos in enumerate(ShotSelection.create_k_folds(i, k)):
        fold[pos] = f
    return (i * k // n_shots).astype(np.int32), fold


def build(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> FoldMoments:
    """One XTC pass -> fold moments on both axes, cached to `cache_path`."""
    from automask.io.read_xtc import iter_calibrated
    from automask.utils import profile_run_values

    profile = profile_run_values(run, show=False)
    indices = selection.resolve(profile)
    if indices.size < 2 * k:
        raise RuntimeError(f"run {run}: {indices.size} shots is too few for {k} folds")
    block_of_shot, fold_of_shot = assign_folds(indices.size, k)
    by_event = dict(zip(indices.tolist(),
                        zip(block_of_shot.tolist(), fold_of_shot.tolist())))

    n = np.zeros(k, dtype=np.float64)
    s1 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    s2 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    dn = np.zeros(k, dtype=np.float64)
    ds1 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    ds2 = np.zeros((k, *PANEL_SHAPE), dtype=np.float64)
    used = 0
    for event_index, panel in iter_calibrated(run, indices, source=profile.source):
        b, f = by_event[event_index]
        frame = panel.astype(np.float64)
        sq = frame * frame
        n[b] += 1
        s1[b] += frame
        s2[b] += sq
        dn[f] += 1
        ds1[f] += frame
        ds2[f] += sq
        used += 1
        if used % 100 == 0:
            print(f"[folds] run {run:04d}: {used}/{indices.size} frames", flush=True)
    if (n < 2).any() or (dn < 2).any():
        raise RuntimeError(f"run {run}: fold counts {n} / {dn} -- a fold got < 2 frames")

    fm = FoldMoments(run=run, n=n, s1=s1, s2=s2, indices=indices,
                     block_of_shot=block_of_shot, dn=dn, ds1=ds1, ds2=ds2,
                     fold_of_shot=fold_of_shot)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(run, k, selection)
    np.savez(path, n=n, s1=s1, s2=s2, dn=dn, ds1=ds1, ds2=ds2, indices=indices,
             block_of_shot=block_of_shot, fold_of_shot=fold_of_shot, run=run)
    print(f"[folds] run {run:04d}: {used} frames over {k} time blocks and "
          f"{k} dealt folds -> {path}")
    return fm


def load(run: int, k: int = N_FOLDS, selection: ShotSelection = LIT) -> FoldMoments:
    """Cached fold moments for `run`, built from XTC on a miss.

    A pre-`LAYOUT` cache has no dealt axis and is not upgradeable in place --
    the deal has to see the frames -- so a miss here means a full rebuild.
    """
    path = cache_path(run, k, selection)
    if not os.path.exists(path):
        return build(run, k, selection)
    z = np.load(path)
    return FoldMoments(run=run, n=z["n"], s1=z["s1"], s2=z["s2"],
                       indices=z["indices"], block_of_shot=z["block_of_shot"],
                       dn=z["dn"], ds1=z["ds1"], ds2=z["ds2"],
                       fold_of_shot=z["fold_of_shot"])


# ==========================================================================
#  a Sample restricted to a subset of shots
# ==========================================================================
def with_shot_images(base, mean_asm: np.ndarray, std_asm: np.ndarray):
    """Copy of Sample `base` with its shot-derived reductions replaced.

    Replaced: `mean` and `std`. Left alone: every calibration constant -- none of
    those is estimated from this run's shots, so resampling shots must not
    perturb them.

    `mean`'s zero set is forced to match the full-run mean: `real` is a geometry
    fact, and letting it flicker with the shot sample would show up as fake
    instability in the geometry floor, which is not what any of these metrics is
    trying to measure.
    """
    mean = np.asarray(mean_asm, dtype=np.float64).copy()
    mean[base.mean == 0] = 0.0
    return base.with_arrays(mean=mean, std=np.asarray(std_asm, dtype=np.float64))


def fold_sample(base, fm: FoldMoments, folds: Sequence[int], dealt: bool = False):
    """Sample `base` rebuilt from the shots in `folds` alone.

    `dealt` picks the axis, as in `FoldMoments.moments`.

    Both sides of a split carry the per-shot mean, so they share an overall
    magnitude: every stat reading it is a robust z and so scale-free anyway, but
    keeping the units fixed means a difference between halves can only come from
    the pixels.
    """
    from automask.geometry import panel_to_asm

    n, mean, std = fm.moments(folds, dealt=dealt)
    return with_shot_images(base, panel_to_asm(mean, fm.run),
                            panel_to_asm(std, fm.run))


def main(runs: Optional[Sequence[int]] = None, k: int = N_FOLDS):
    from automask.evaluation import EVAL_RUNS
    for run in (EVAL_RUNS if runs is None else runs):
        fm = load(run, k)
        n, mean, std = fm.moments(range(fm.k))
        nd, _, _ = fm.moments(range(fm.k), dealt=True)
        print(f"  run {run}: {fm.k} time blocks {fm.n.astype(int).tolist()}")
        print(f"            {fm.k} dealt folds {fm.dn.astype(int).tolist()}")
        print(f"            {int(n)} shots on both axes ({'consistent' if n == nd else 'MISMATCH'}), "
              f"pooled mean {mean.mean():.3f}, pooled std {std.mean():.3f}")


if __name__ == "__main__":
    import sys
    main([int(a) for a in sys.argv[1:]] or None)
