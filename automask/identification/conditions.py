"""
identification/conditions.py -- what "experimental condition" means for a run,
and the one XTC pass that reduces the run to per-condition per-pixel means.

The whole contrast argument rests on the sample signal MOVING between groups of
shots: `S(q; theta_c)` must differ across `c` or nothing separates `tau_i` from
`S`. So the first question is not "how do we estimate tau" but "what is `c`
here, and does it move the signal at all". This module supplies the candidate
answers and leaves the measurement to
`studies/loss_identification.py::exp_m2_identifying_power`.

FOUR CANDIDATE CONDITION AXES, in decreasing order of how much they should move
the sample and increasing order of how much new plumbing they need:

  `delay`  -- the CO2 delay-scan motor (`epicsAll/delay` in small data). This is
              the axis the experiment was actually scanning, so it is the one
              that should carry real signal change. It is NOT in `ShotMeta`;
              `scan_epics` below reads it straight off the EPICS store, and
              returns None on a run that does not publish it.
  `branch` -- CC vs VCC. One beam split two ways (see CLAUDE.md); the two
              branches traverse different optics, so `S` may differ. Free: the
              shutter voltages are already in `ShotMeta`.
  `time`   -- contiguous blocks of the run. In a scan run this is a coarse proxy
              for `delay`; in a static run it measures drift instead, which is
              why it doubles as the test of assumption 1.
  `flux`   -- quantiles of a downstream monitor. The WEAKEST axis by
              construction: the model divides `F_t` out, so flux groups differ
              in `S` only through nonlinearity of the sample response. It is
              included precisely because it should score near zero identifying
              power -- it is the negative control for the whole idea.

WHY THE MOMENTS ARE ACCUMULATED FLUX-NORMALISED. The estimator needs
`zbar_i^(c) = mean_t in c (x_it / F_t)`, not `mean(x) / mean(F)`: `F_t` varies
shot to shot by tens of percent and the two differ at first order in
`Var(F)/E[F]^2`. Normalising inside the accumulation costs nothing and makes the
group mean an unbiased estimate of `tau_i A_i S(q_i; theta_c) + J_i` exactly as
written. Second moments are accumulated on the same normalised quantity, so
`var / n` is the honest sampling variance of the group mean and can be used as
the per-pixel noise model downstream.

The linear domain, not the log domain: calibrated Jungfrau ADU are frequently
negative in the low-signal corners (pedestal subtraction is two-sided), so
`log` is undefined on a large minority of pixels. The two-way-fixed-effects
picture in the notes is a way to SEE the identifiability structure; the
estimators are run in the linear domain, where the contrast cancels `J_i`
exactly rather than approximately.

Run:  python -m automask.identification.conditions 475 --scheme delay
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from automask.shot_selection import ShotSelection

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PKG, "outputs", "cache", "conditions")
PANEL_SHAPE = (2, 512, 1024)

#: The lit-beam population every identification estimator works on. Identical to
#: `unsupervised.folds.LIT` on purpose: condition groups must partition the SAME
#: shots the pipeline is fed, or the estimate describes a different run.
LIT = ShotSelection(beam="on")

#: EPICS PV aliases to try for the delay-scan motor, in order. Small data stores
#: the first as `epicsAll/delay` (`xpp_sharing/utils.py` reads it there).
DELAY_PVS = ("delay", "lxt", "lxt_ttc", "las_fs14_target_time")

SCHEMES = ("delay", "branch", "time", "flux")


# ==========================================================================
#  condition labels -- pure numpy, no psana
# ==========================================================================
@dataclass
class Conditions:
    """A partition of selected shots into condition groups.

    `labels` is one group id per entry of `indices` (event indices in stream
    order, as `ShotSelection.resolve` returns them). Group ids are contiguous
    from 0. `values` carries the axis value each group was built from, for
    reporting -- a delay in motor units, a flux quantile midpoint, 0/1 for the
    branch -- and is NaN where the axis has no numeric meaning.
    """
    scheme: str
    indices: np.ndarray
    labels: np.ndarray
    values: np.ndarray
    note: str = ""

    @property
    def n_groups(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.size else 0

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_groups)


def _quantile_bins(x: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-count binning of `x` into `n_groups`, with group medians.

    Equal-count rather than equal-width for the same reason the azimuthal rings
    are: a group's sampling noise scales as 1/sqrt(n), and unequal groups make
    the identifying-power estimate a function of the binning instead of the
    physics.
    """
    order = np.argsort(x, kind="stable")
    labels = np.empty(x.size, dtype=np.int32)
    edges = (np.arange(x.size) * n_groups) // x.size
    labels[order] = edges
    values = np.array([np.median(x[labels == g]) if (labels == g).any() else np.nan
                       for g in range(n_groups)])
    return labels, values


def label_shots(meta, indices: np.ndarray, scheme: str = "time",
                n_groups: int = 4, delay: Optional[np.ndarray] = None,
                monitor: str = "sample_diode") -> Conditions:
    """Partition `indices` into condition groups under `scheme`.

    `delay` is the per-event motor readback from `scan_epics`, required by
    `scheme="delay"` and ignored otherwise. Shots whose axis value is not finite
    are dropped from `indices` rather than pooled into a junk group.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if scheme == "delay":
        if delay is None:
            raise ValueError("scheme='delay' needs the motor readback; call "
                             "scan_epics(run) first (returns None if the run "
                             "does not publish a delay PV)")
        x = np.asarray(delay, dtype=np.float64)[indices]
        keep = np.isfinite(x)
        indices, x = indices[keep], x[keep]
        uniq = np.unique(x)
        if uniq.size <= n_groups:
            labels = np.searchsorted(uniq, x).astype(np.int32)
            return Conditions("delay", indices, labels, uniq,
                              note=f"{uniq.size} distinct motor positions")
        labels, values = _quantile_bins(x, n_groups)
        return Conditions("delay", indices, labels, values,
                          note=f"{uniq.size} positions binned into {n_groups}")

    if scheme == "branch":
        cc = np.asarray(meta.cc_open, bool)[indices]
        vcc = np.asarray(meta.vcc_open, bool)[indices]
        code = cc.astype(np.int32) + 2 * vcc.astype(np.int32)
        present = np.unique(code)
        labels = np.searchsorted(present, code).astype(np.int32)
        names = {0: "neither", 1: "CC", 2: "VCC", 3: "both"}
        return Conditions("branch", indices, labels,
                          present.astype(np.float64),
                          note="/".join(names[int(c)] for c in present))

    if scheme == "time":
        labels = ((np.arange(indices.size) * n_groups) // indices.size).astype(np.int32)
        return Conditions("time", indices, labels,
                          np.arange(n_groups, dtype=np.float64),
                          note=f"{n_groups} contiguous blocks in stream order")

    if scheme == "flux":
        x = meta.monitor(monitor)[indices]
        keep = np.isfinite(x) & (x != 0)
        indices, x = indices[keep], x[keep]
        labels, values = _quantile_bins(x, n_groups)
        return Conditions("flux", indices, labels, values,
                          note=f"{monitor} quantiles (negative control)")

    raise ValueError(f"unknown scheme {scheme!r}; known: {list(SCHEMES)}")


# ==========================================================================
#  per-condition moments -- one XTC pass
# ==========================================================================
@dataclass
class GroupMoments:
    """Flux-normalised per-pixel `(count, sum, sum of squares)` per condition.

    Arrays are native panel space `(G, 2, 512, 1024)`. The accumulated quantity
    is `z_it = x_it / F_t`, so `mean()` estimates
    `tau_i A_i S(q_i; theta_c) + J_i` with no flux term left in it.
    """
    run: int
    scheme: str
    n: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    values: np.ndarray
    flux: np.ndarray                 # mean F_t per group, for reporting
    monitor: str

    @property
    def n_groups(self) -> int:
        return int(self.n.size)

    def mean(self, groups: Optional[Sequence[int]] = None) -> np.ndarray:
        """Per-pixel group mean, `(G, ...)` or pooled over `groups`."""
        if groups is None:
            return self.s1 / self.n[:, None, None, None]
        g = np.asarray(list(groups), dtype=int)
        return self.s1[g].sum(axis=0) / self.n[g].sum()

    def sem2(self) -> np.ndarray:
        """Squared standard error of each group mean, `(G, ...)`.

        This is the noise model every downstream test uses. It is measured, not
        assumed Poisson: common mode and gain-stage switching both break the
        Poisson relation between mean and variance on this detector.
        """
        n = self.n[:, None, None, None]
        var = np.maximum(self.s2 / n - (self.s1 / n) ** 2, 0.0)
        return var / np.maximum(n - 1, 1)


def cache_path(run: int, scheme: str, n_groups: int,
               selection: ShotSelection = LIT) -> str:
    from automask.features.base import FeatureSpec
    key = FeatureSpec("_", "mean", selection).content_key
    return os.path.join(CACHE_DIR,
                        f"cond_{key}_{scheme}_g{n_groups}_run{run:04d}.npz")


def scan_epics(run: int, pvs: Sequence[str] = DELAY_PVS) -> Optional[np.ndarray]:
    """Per-event readback of the first PV in `pvs` the run publishes.

    Returns None (not an exception) when no candidate PV exists: whether this
    experiment's runs carry a scannable condition axis at all is one of the open
    questions, and "the axis is absent" is an answer the study has to be able to
    report.
    """
    import psana
    from automask.io.read_xtc import open_local_run

    ds, _ = open_local_run(run)
    store = ds.env().epicsStore()
    name = next((p for p in pvs if store.value(p) is not None), None)
    if name is None:
        print(f"[epics] run {run}: none of {list(pvs)} is published")
        return None
    out = []
    for evt in ds.events():
        v = ds.env().epicsStore().value(name)
        out.append(float(v) if v is not None else np.nan)
    print(f"[epics] run {run}: PV {name!r}, {np.unique(np.round(out, 6)).size} "
          f"distinct values over {len(out)} events")
    return np.asarray(out, dtype=np.float64)


def build(run: int, scheme: str = "time", n_groups: int = 4,
          selection: ShotSelection = LIT,
          monitor: str = "sample_diode") -> GroupMoments:
    """One XTC pass -> flux-normalised per-condition moments, cached."""
    from automask.io.read_xtc import iter_calibrated, scan_shots

    meta = scan_shots(run)
    indices = selection.resolve(meta)
    delay = scan_epics(run) if scheme == "delay" else None
    cond = label_shots(meta, indices, scheme, n_groups, delay=delay,
                       monitor=monitor)
    flux_all = meta.monitor(monitor)
    if not np.isfinite(flux_all[cond.indices]).all():
        raise RuntimeError(f"run {run}: monitor {monitor!r} has non-finite "
                           f"readings inside the selection")

    g_of_event = dict(zip(cond.indices.tolist(), cond.labels.tolist()))
    G = cond.n_groups
    n = np.zeros(G)
    s1 = np.zeros((G, *PANEL_SHAPE))
    s2 = np.zeros((G, *PANEL_SHAPE))
    fsum = np.zeros(G)
    used = 0
    for event_index, panel in iter_calibrated(run, cond.indices):
        g = g_of_event[event_index]
        z = panel.astype(np.float64) / flux_all[event_index]
        n[g] += 1
        s1[g] += z
        s2[g] += z * z
        fsum[g] += flux_all[event_index]
        used += 1
        if used % 100 == 0:
            print(f"[cond] run {run:04d}: {used}/{cond.indices.size} frames",
                  flush=True)
    if (n < 2).any():
        raise RuntimeError(f"run {run}: group counts {n} -- a group got < 2 frames")

    gm = GroupMoments(run=run, scheme=scheme, n=n, s1=s1, s2=s2,
                      values=cond.values, flux=fsum / n, monitor=monitor)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(run, scheme, n_groups, selection)
    np.savez(path, n=n, s1=s1, s2=s2, values=cond.values, flux=gm.flux,
             run=run, scheme=scheme, monitor=monitor)
    print(f"[cond] run {run:04d}: {used} frames over {G} {scheme} groups "
          f"({cond.note}) -> {path}")
    return gm


def load(run: int, scheme: str = "time", n_groups: int = 4,
         selection: ShotSelection = LIT, monitor: str = "sample_diode"):
    path = cache_path(run, scheme, n_groups, selection)
    if not os.path.exists(path):
        return build(run, scheme, n_groups, selection, monitor)
    z = np.load(path, allow_pickle=False)
    return GroupMoments(run=run, scheme=str(z["scheme"]), n=z["n"], s1=z["s1"],
                        s2=z["s2"], values=z["values"], flux=z["flux"],
                        monitor=str(z["monitor"]))


def describe(run: int, schemes: Sequence[str] = SCHEMES,
             n_groups: int = 4) -> Dict[str, Conditions]:
    """Label the run under every scheme WITHOUT touching the frames.

    Cheap enough to run first: it reports how many shots each axis actually
    resolves into how many groups, which decides whether the expensive pass is
    worth making at all.

    Goes through `automask.io.scan_shots`, so it runs without psana -- the
    survey needs only per-shot scalars, which `io.xtc_raw` decodes directly.
    The `delay` axis is the exception: it lives in the EPICS store, which the
    raw reader does not decode, so it reports as unavailable rather than absent.
    """
    from automask.io import scan_shots

    meta = scan_shots(run)
    indices = LIT.resolve(meta)
    delay = None
    out = {}
    for scheme in schemes:
        if scheme == "delay" and delay is None:
            try:
                delay = scan_epics(run)
            except ImportError:
                print(f"  {scheme:<8} unavailable (needs psana for the EPICS store)")
                continue
            if delay is None:
                print(f"  {scheme:<8} unavailable (no delay PV)")
                continue
        try:
            c = label_shots(meta, indices, scheme, n_groups, delay=delay)
        except Exception as e:                    # noqa: BLE001 -- reported
            print(f"  {scheme:<8} failed: {type(e).__name__}: {e}")
            continue
        out[scheme] = c
        print(f"  {scheme:<8} {c.n_groups} groups, counts "
              f"{c.counts.tolist()}  ({c.note})")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=int, nargs="*", default=None)
    ap.add_argument("--scheme", default=None, choices=list(SCHEMES))
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args()
    from automask.evaluation import EVAL_RUNS
    for run in (args.runs or list(EVAL_RUNS)):
        print(f"\nrun {run}:")
        if args.scheme is None:
            describe(run, n_groups=args.groups)
        else:
            gm = load(run, args.scheme, args.groups)
            print(f"  {gm.n_groups} groups, n {gm.n.astype(int).tolist()}, "
                  f"mean flux {np.round(gm.flux, 4).tolist()}")


if __name__ == "__main__":
    main()
