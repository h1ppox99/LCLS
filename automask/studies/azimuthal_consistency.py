#!/usr/bin/env python3
"""
studies/azimuthal_consistency.py -- the azimuthal-isotropy metric, explored.

The statistic itself (excess azimuthal scatter beyond the noise floor, the
random-mask control, the winsorization and radial-gradient-leakage systematics)
now lives in `automask.unsupervised.azimuthal`, which documents WHY it is built
the way it is -- in particular why an ANOVA F is the wrong instrument here. This
file is the driver that got it there and is kept for what a metric module should
not carry: the per-detector variant table, the sector-count sensitivity sweep,
the per-ring curves, and the Welch F column that shows the significance test
failing.

For the question "does this metric rank masks the way the human reference does",
see `studies/metric_validation.py`.

Run:  python -m automask.studies.azimuthal_consistency
      python -m automask.studies.azimuthal_consistency --runs 475 --sectors 12
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automask.evaluation import EVAL_RUNS, load_sample
from automask.masking import production_pipeline
from automask.unsupervised.azimuthal import (
    CLIP_Q, MIN_SECTORS, N_MIN, TARGET_CELL, cell_moments, excess_scatter,
    gradient_leakage, pixel_frame, ring_edges, ring_reference,
    sector_edges_per_ring, winsorize_per_ring,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

# ==========================================================================
#  mask variants
# ==========================================================================
def variants(run):
    prod = production_pipeline()
    sample = load_sample(
        run, selection=prod.shot_selection, reductions=prod.reductions_needed(),
        calibrations=prod.calibrations_needed(),
    )
    floor = prod.floor(sample)
    names = [d.stat for d in prod.detectors]
    picks = {d.stat: d.pick(sample) for d in prod.detectors}
    out = {"unmasked": np.zeros_like(floor), "floor only": floor.copy()}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            m = floor.copy()
            for c in combo:
                m = m | picks[c]
            out["floor+" + "+".join(c[:4] for c in combo)] = m
    return out, floor


def random_control(mask, floor, valid, rng):
    extra = int((mask & ~floor).sum())
    pool = np.flatnonzero((~floor & valid).ravel())
    if extra == 0 or pool.size == 0:
        return floor.copy()
    chosen = rng.choice(pool, size=min(extra, pool.size), replace=False)
    out = floor.copy().ravel()
    out[chosen] = True
    return out.reshape(mask.shape)


# ==========================================================================
#  study
# ==========================================================================
def run_study(run: int, n_sectors: int = 12, sector_sweep=(4, 8, 16, 32), rng=None):
    rng = rng or np.random.default_rng(0)
    q, chi, inten, valid = pixel_frame(
        load_sample(run, reductions=(), calibrations=())
    )
    masks, floor = variants(run)
    usable = valid & np.isfinite(inten)

    n_rings = max(int(usable.sum() / (TARGET_CELL * n_sectors)), 8)
    edges = ring_edges(q, usable, n_rings)
    n_rings = len(edges) - 1
    ring_idx = np.clip(np.digitize(q, edges) - 1, 0, n_rings - 1)
    sec_idx = sector_edges_per_ring(ring_idx, chi, usable & ~floor, n_rings, n_sectors)
    clipped = winsorize_per_ring(inten, ring_idx, usable, n_rings)

    print(f"\n{'=' * 110}")
    print(f"RUN {run}   {n_rings} equal-count rings x {n_sectors} equal-count sectors "
          f"({TARGET_CELL} px/cell target)")
    print("=" * 110)
    print(f"{'mask':<22} {'kept%':>7} | {'median':>7} {'mean':>7} {'p90':>7} | "
          f"{'gain':>7} {'win%':>6} {'sign p':>9} | {'Welch F':>8}")
    print(f"{'':<22} {'':>7} | {'--- excess scatter % ---':^23} | "
          f"{'------ vs random control ------':^24} | {'':>8}")

    # Per-ring (sigma^2, mu) from the floor-only pixels: one reference per
    # intensity field, shared by every mask below and by the leakage floor.
    ref = {label: ring_reference(*cell_moments(ring_idx, sec_idx, field,
                                               usable & ~floor, n_rings, n_sectors))
           for label, field in (("raw", inten), ("win", clipped))}

    rows = {}
    for tag, m in masks.items():
        keep = usable & ~m
        ctrl = random_control(m, floor, usable, rng)
        keep_c = usable & ~ctrl
        res = {}
        for label, field in (("raw", inten), ("win", clipped)):
            e, mu, F = excess_scatter(*cell_moments(ring_idx, sec_idx, field, keep,
                                                    n_rings, n_sectors),
                                      ref=ref[label])
            ec, _, _ = excess_scatter(*cell_moments(ring_idx, sec_idx, field, keep_c,
                                                    n_rings, n_sectors),
                                      ref=ref[label])
            res[label] = (np.nanmedian(e), np.nanmedian(ec), e, mu, np.nanmedian(F),
                          np.nanmean(e), np.nanpercentile(e, 90), ec)
        leak, qmid = gradient_leakage(ring_idx, sec_idx, q, keep,
                                      n_rings, n_sectors, ref["raw"][1])
        r1 = res["win"]
        # paired per-ring comparison against the equally-sized random mask
        both = np.isfinite(r1[2]) & np.isfinite(r1[7])
        # A detector that adds nothing beyond the floor has an IDENTICAL control,
        # so the comparison is undefined rather than a loss.
        if int((m & ~floor).sum()) == 0:
            both = np.zeros_like(both)
        n_pair = int(both.sum())
        n_win = int((r1[2][both] < r1[7][both]).sum())
        better = n_win / n_pair if n_pair else np.nan
        # THE test: a paired sign test over rings. Absolute p-values from the
        # ANOVA are meaningless (the sample is genuinely anisotropic, so H0 is
        # false everywhere), but the ring-by-ring comparison against an
        # equally-sized RANDOM mask conditions on that same anisotropy. H0 here
        # is "the mask does no better than dropping the same number of pixels at
        # random", under which each ring is a coin flip.
        pval = stats.binomtest(n_win, n_pair, 0.5).pvalue if n_pair else np.nan
        if not n_pair:
            better = np.nan
        rows[tag] = dict(kept=keep.sum() / usable.sum(), res=res, qmid=qmid,
                         leak=np.nanmedian(leak), curve=r1[2], better=better,
                         pval=pval, n_pair=n_pair)
        print(f"{tag:<22} {100*rows[tag]['kept']:6.2f}% | "
              f"{100*r1[0]:6.3f}% {100*r1[5]:6.3f}% {100*r1[6]:6.3f}% | "
              f"{100*(r1[1]-r1[0]):+6.3f}% " +
              (f"{100*better:5.1f}% {pval:9.2e}" if n_pair else f"{'n/a':>5} {'n/a':>9}") + " | " +
              f"{r1[4]:8.1f}")

    print(f"\n  radial-gradient leakage floor (median): "
          f"{100*rows['floor only']['leak']:.4f}%  "
          f"-- `excess` below this is binning bias, not physics")

    print(f"\n  sector-count sensitivity (winsorized excess%, floor only / all 3):")
    prod_tag = [t for t in masks if t.count("+") == 3]
    prod_tag = prod_tag[0] if prod_tag else list(masks)[-1]
    sweep = {}
    for S in sector_sweep:
        si = sector_edges_per_ring(ring_idx, chi, usable & ~floor, n_rings, S)
        ref_S = ring_reference(*cell_moments(ring_idx, si, clipped,
                                             usable & ~floor, n_rings, S))
        vals = []
        for tag in ("floor only", prod_tag):
            e, _, _ = excess_scatter(*cell_moments(ring_idx, si, clipped,
                                                   usable & ~masks[tag], n_rings, S),
                                     ref=ref_S)
            vals.append(np.nanmedian(e))
        lk, _ = gradient_leakage(ring_idx, si, q, usable & ~floor,
                                 n_rings, S, ref_S[1])
        sweep[S] = vals + [np.nanmedian(lk)]
        print(f"    S={S:<3d} {usable.sum()/(n_rings*S):6.0f} px/cell   "
              f"{100*vals[0]:7.3f}%   {100*vals[1]:7.3f}%   "
              f"(leakage floor {100*np.nanmedian(lk):.4f}%)")

    return dict(run=run, rows=rows, sweep=sweep, sector_sweep=sector_sweep,
                prod_tag=prod_tag, n_sectors=n_sectors,
                qmid=rows["floor only"]["qmid"])


def plot_study(res, out_path):
    rows, qmid = res["rows"], res["qmid"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    order = list(rows)
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i / max(len(order) - 1, 1)) for i, t in enumerate(order)}

    a = ax[0, 0]
    for tag in order:
        c = rows[tag]["curve"]
        k = max(len(c) // 60, 1)
        sm = np.array([np.nanmedian(c[i:i + k]) for i in range(0, len(c), k)])
        qs = np.array([np.nanmedian(qmid[i:i + k]) for i in range(0, len(qmid), k)])
        a.plot(qs, 100 * sm, lw=1.4, label=tag, color=colors[tag], alpha=0.9)
    a.axhline(100 * rows["floor only"]["leak"], color="r", ls=":", lw=1.2,
              label="gradient-leakage floor")
    a.set_yscale("log")
    a.set_xlabel("q  [$\\AA^{-1}$]")
    a.set_ylabel("excess azimuthal scatter  [% of ring mean]")
    a.set_title(f"run {res['run']}: azimuthal inhomogeneity vs q (winsorized)\n"
                "lower = sectors agree = mask is working", fontsize=10)
    a.legend(fontsize=6.5, ncol=2)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for tag in order:
        r = rows[tag]
        a.scatter(100 * r["kept"], 100 * r["res"]["win"][0], s=80,
                  color=colors[tag], zorder=3)
        a.scatter(100 * r["kept"], 100 * r["res"]["win"][1], s=45,
                  color=colors[tag], marker="x", zorder=3)
        a.annotate(tag, (100 * r["kept"], 100 * r["res"]["win"][0]), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("pixels retained  [%]")
    a.set_ylabel("median excess scatter  [%]")
    a.set_title("consistency vs information kept\n"
                "dot = real mask, x = random mask of identical size", fontsize=10)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    S = res["sector_sweep"]
    a.plot(S, [100 * res["sweep"][s][0] for s in S], "o-", label="floor only")
    a.plot(S, [100 * res["sweep"][s][1] for s in S], "s-", label=res["prod_tag"])
    a.plot(S, [100 * res["sweep"][s][2] for s in S], ":", color="r",
           label="leakage floor")
    a.set_xscale("log", base=2)
    a.set_xlabel("number of azimuthal sectors S")
    a.set_ylabel("median excess scatter  [%]")
    a.set_title("sector-count sensitivity\nfewer sectors = coarser defects, "
                "more px/cell", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    base = rows["floor only"]["res"]["win"][0]
    tags = [t for t in order if t != "unmasked"]
    gains = [100 * (base - rows[t]["res"]["win"][0]) for t in tags]
    ctrl = [100 * (base - rows[t]["res"]["win"][1]) for t in tags]
    y = np.arange(len(tags))
    a.barh(y - 0.2, gains, height=0.4, label="real mask")
    a.barh(y + 0.2, ctrl, height=0.4, label="random mask, same size")
    a.set_yticks(y)
    a.set_yticklabels(tags, fontsize=8)
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("improvement in excess scatter vs floor-only  [pp]")
    a.set_title("does each detector earn its pixels?", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Azimuthal consistency as a label-free mask score -- run {res['run']}"
                 f"  ({res['n_sectors']} sectors)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="*", default=list(EVAL_RUNS))
    ap.add_argument("--sectors", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for run in args.runs:
        res = run_study(run, n_sectors=args.sectors)
        plot_study(res, os.path.join(OUT_DIR,
                                     f"azimuthal_consistency_run{run:04d}.png"))


if __name__ == "__main__":
    main()
