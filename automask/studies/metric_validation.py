#!/usr/bin/env python3
"""
studies/metric_validation.py -- are the label-free metrics actually metrics?

`automask.unsupervised` proposes twelve surrogates for a quantity production
cannot measure: how good is this mask? This study asks the only question that
makes any of them worth running -- on the two runs where a human reference DOES
exist, does each surrogate rank masks the way the truth does?

METHOD. Build a panel of candidate masks spanning the failure modes that matter
(under-masking, over-masking, structured-but-wrong, random-but-right-sized, plus
the real pipeline and its knob variants), score every candidate with every
unsupervised metric, and correlate each metric against `IoU vs human` across the
panel. Spearman rather than Pearson: a metric only has to get the ORDER right,
and nothing here is expected to be linear in IoU.

THREE THINGS THE HEADLINE NUMBER HIDES, all reported separately.

  * EASY PANEL vs HARD PANEL. A rank correlation computed over a panel that
    includes deliberate garbage mostly measures "can you spot garbage", which is
    not the production question. `rho_top` repeats the correlation over the top
    half of the panel by true IoU -- discrimination among masks that are all
    roughly reasonable, which is what selecting a recipe actually requires.
  * TWO TARGETS. `IoU vs human` is dominated by the geometry+calib floor, which
    every sane candidate gets right for free. `residual IoU` (the pixels beyond
    the floor, against the human pixels beyond the floor) is the part the
    intensity detectors are responsible for, and is the harder target.
  * PER-RUN AGREEMENT. Two runs is a small n, but a metric whose rho flips sign
    between 389 and 475 is not measuring a property of masks.

Run:  python -m automask.studies.metric_validation
      python -m automask.studies.metric_validation --quick --runs 475
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automask.dataset import score
from automask.evaluation import EVAL_RUNS, load_sample
from automask.masking import Detector, Pipeline, production_pipeline
from automask.unsupervised import METRICS, Candidate, MetricContext, score_candidate
from automask.unsupervised.stability import pipeline_jitter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

# The features every candidate's stats need, plus `umean` for the analytic-noise
# metric. Asked for explicitly so a cold cache builds once, not per candidate.
FEATURES = ("ustd", "umean", "pedestal")


# ==========================================================================
#  the candidate panel
# ==========================================================================
def _pipe(k=3.5, tv=4.0, dets=("variance", "hough", "asic"), combiner="union"):
    """A production-shaped pipeline with the two most consequential knobs free."""
    from automask.stats.variance import VarianceParams
    from automask.stats.hough_lines import HoughLinesParams
    from automask.stats.asic_polish import AsicPolishParams
    from automask.regularization.tv import TVParams
    from automask.regularization.blob_scale import BlobScaleParams
    from automask.regularization.fill_holes import FillHolesParams
    from automask.regularization.area_gate import AreaGateParams
    from automask.combine.weighted_sum import WeightedSumParams

    d = []
    if "variance" in dets:
        d.append(Detector("variance", VarianceParams(k=k, mode="low"),
                          field_reg="tv", field_reg_params=TVParams(tv)))
    if "hough" in dets:
        d.append(Detector("hough_lines", HoughLinesParams(defectiveness_scale=5.0),
                          field_reg=None))
    if "asic" in dets:
        d.append(Detector("asic_polish",
                          AsicPolishParams(asic=256, n_iter=3, k=15.0, mode="high"),
                          field_reg=["blob_scale"], field_reg_params=[BlobScaleParams()],
                          mask_reg=["fill_holes", "area_gate"],
                          mask_reg_params=[FillHolesParams(), AreaGateParams()]))
    params = WeightedSumParams(k=3.5, pad=2) if combiner == "weighted_sum" else None
    return Pipeline(d, combiner=combiner, combiner_params=params)


def _morph(pipe, op, radius):
    """Production mask grown or shrunk -- the two pure failure directions, with
    everything else about the recipe held fixed."""
    from skimage.morphology import binary_dilation, binary_erosion, disk
    f = binary_dilation if op == "dilate" else binary_erosion
    return lambda s: f(pipe.run(s), disk(radius))


def _random_like(pipe, blobs: bool):
    """A mask of the RIGHT SIZE in the WRONG PLACES.

    The control the whole package leans on, promoted here to a candidate: any
    metric that cannot separate this from the real mask is measuring how many
    pixels were dropped, not which. `blobs=True` additionally gives it the right
    MORPHOLOGY (random disks rather than salt-and-pepper), which is what
    separates tier-0 compactness from the physics tiers.

    Deliberately reseeded on every call, so the reproducibility metrics see it
    for what it is.
    """
    from skimage.morphology import disk

    def make(s):
        floor = production_pipeline().floor(s)
        m = pipe.run(s)
        want = int((m & ~floor).sum())
        rng = np.random.default_rng()
        out = floor.copy()
        pool = np.flatnonzero((~floor & s.real).ravel())
        if not blobs:
            pick = rng.choice(pool, size=min(want, pool.size), replace=False)
            flat = out.ravel(); flat[pick] = True
            return flat.reshape(m.shape)
        d = disk(3)
        per = int(d.sum())
        flat = out.ravel()
        for _ in range(max(want // per, 1)):
            c = int(rng.choice(pool))
            r0, c0 = np.unravel_index(c, m.shape)
            rr, cc = np.nonzero(d)
            rr = np.clip(rr + r0 - 3, 0, m.shape[0] - 1)
            cc = np.clip(cc + c0 - 3, 0, m.shape[1] - 1)
            out2 = flat.reshape(m.shape)
            out2[rr, cc] = True
        return flat.reshape(m.shape) & (s.real | floor)
    return make


def build_panel(quick: bool = False):
    """The candidates, ordered roughly worst-to-best by intent (not by score --
    the ordering is never used, only reported against)."""
    prod = production_pipeline()
    cands = [
        Candidate("floor_only", _pipe(dets=()).run, jitter=pipeline_jitter(_pipe(dets=())),
                  note="under-mask: geometry+calib only"),
        Candidate("production", prod.run, jitter=pipeline_jitter(prod),
                  note="the default recipe"),
        Candidate("random_px", _random_like(prod, blobs=False),
                  note="right size, wrong pixels, no structure"),
        Candidate("random_blobs", _random_like(prod, blobs=True),
                  note="right size and morphology, wrong places"),
        Candidate("dilate_r5", _morph(prod, "dilate", 5), note="over-mask"),
    ]
    if quick:
        return cands

    for k in (2.0, 2.5, 3.0, 4.5, 6.0):
        p = _pipe(k=k)
        cands.append(Candidate(f"var_k{k}", p.run, jitter=pipeline_jitter(p),
                               note=f"variance threshold {k}"))
    for tv in (0.5, 16.0):
        p = _pipe(tv=tv)
        cands.append(Candidate(f"tv{tv}", p.run, jitter=pipeline_jitter(p),
                               note=f"TV weight {tv}"))
    for dets in (("variance",), ("hough",), ("asic",), ("variance", "hough")):
        p = _pipe(dets=dets)
        cands.append(Candidate("det_" + "+".join(x[:4] for x in dets), p.run,
                               jitter=pipeline_jitter(p), note="detector subset"))
    p = _pipe(combiner="weighted_sum")
    cands.append(Candidate("weighted_sum", p.run, jitter=pipeline_jitter(p),
                           note="fusion instead of union"))
    cands += [
        Candidate("dilate_r2", _morph(prod, "dilate", 2), note="mild over-mask"),
        Candidate("erode_r2", _morph(prod, "erode", 2), note="mild under-mask"),
        Candidate("human_ref", lambda s: s.human, static=True,
                  note="the ground truth itself -- a metric that does not rank "
                       "this near the top is suspect"),
    ]
    try:                                     # the lab's own production mask
        from automask.dataset import load_mask
        load_mask("cmask_run0475")
        cands.append(Candidate(
            "lab_cmask", lambda s: load_mask(f"cmask_run{s.run:04d}"), static=True,
            note="the lab's manual production mask"))
    except Exception as e:                   # noqa: BLE001
        print(f"  [note] lab mask unavailable, skipped: {e}")
    return cands


# ==========================================================================
#  scoring
# ==========================================================================
def score_run(run: int, cands, seed: int = 0) -> dict:
    sample = load_sample(run, features=FEATURES)
    ctx = MetricContext(sample=sample, seed=seed)
    floor = ctx.floor()
    target = sample.human & ~floor
    rows = {}
    for c in cands:
        m = c.mask(ctx)
        row = score_candidate(c, ctx)
        row["truth_iou"] = score(m, sample.human)["iou"]
        row["truth_residual"] = score(m & ~floor, target)["iou"]
        row["masked_frac"] = float(m.mean())
        rows[c.name] = row
        print(f"  {c.name:16s} IoU {row['truth_iou']:.3f} "
              f"resid {row['truth_residual']:.3f} "
              f"({100*row['masked_frac']:5.2f}% masked)", flush=True)
    return rows


def rank_table(per_run: dict, target: str = "truth_iou") -> dict:
    """Spearman rho of every metric against `target`, per run and pooled.

    Pooling is done on WITHIN-RUN ranks, not raw values: the two runs sit at
    different absolute levels on several metrics (different beam, different
    defects), and pooling raw values would let that offset masquerade as
    correlation.
    """
    names = list(METRICS)
    runs = list(per_run)
    out = {}
    for name in names:
        spec = METRICS[name]
        sign = 1.0 if spec.higher_is_better else -1.0
        per, top, pooled_x, pooled_y = {}, {}, [], []
        for run, rows in per_run.items():
            cands = [c for c in rows if np.isfinite(rows[c].get(name, np.nan))]
            if len(cands) < 4:
                continue
            x = sign * np.array([rows[c][name] for c in cands])
            y = np.array([rows[c][target] for c in cands])
            if np.std(x) == 0:
                # Not a failure: a metric every candidate satisfies identically
                # (every sane mask contains the floor) carries no ranking
                # information here, and saying so beats reporting a NaN.
                per[run] = {"rho": float("nan"), "p": float("nan"),
                            "n": len(cands), "constant": True}
                continue
            rho, p = stats.spearmanr(x, y)
            per[run] = {"rho": float(rho), "p": float(p), "n": len(cands)}
            # discrimination among the good half only
            cut = np.median(y)
            sel = y >= cut
            if sel.sum() >= 4:
                r2, p2 = stats.spearmanr(x[sel], y[sel])
                top[run] = {"rho": float(r2), "p": float(p2), "n": int(sel.sum())}
            pooled_x.append(stats.rankdata(x) / len(x))
            pooled_y.append(stats.rankdata(y) / len(y))
        if not per:
            out[name] = {"per_run": {}, "pooled_rho": float("nan"),
                         "top_rho": float("nan"), "tier": spec.tier}
            continue
        prho, pp = stats.spearmanr(np.concatenate(pooled_x), np.concatenate(pooled_y))
        out[name] = {
            "per_run": per, "top_per_run": top,
            "pooled_rho": float(prho), "pooled_p": float(pp),
            "top_rho": float(np.mean([top[r]["rho"] for r in top])) if top else float("nan"),
            "tier": spec.tier,
            "consistent": bool(len(per) == len(runs) and
                               len({np.sign(per[r]["rho"]) for r in per}) == 1),
        }
    return out


def composite(per_run: dict) -> dict:
    """Per-tier and overall score cards: the mean z-score of the oriented metrics
    within a run. Z-scoring within the run is what makes metrics on wildly
    different scales (an IoU, a percentage, a win rate) addable at all."""
    out = {}
    for run, rows in per_run.items():
        cands = list(rows)
        tiers = {}
        for name, spec in METRICS.items():
            v = np.array([rows[c].get(name, np.nan) for c in cands], dtype=float)
            if not np.isfinite(v).any() or np.nanstd(v) == 0:
                continue
            z = (v - np.nanmean(v)) / np.nanstd(v)
            tiers.setdefault(spec.tier, []).append(
                z if spec.higher_is_better else -z)
        per_tier = {t: np.nanmean(np.vstack(v), axis=0) for t, v in tiers.items()}
        overall = np.nanmean(np.vstack(list(per_tier.values())), axis=0)
        out[run] = {"candidates": cands,
                    "tiers": {int(t): v.tolist() for t, v in per_tier.items()},
                    "overall": overall.tolist()}
    return out


# ==========================================================================
#  reporting
# ==========================================================================
def print_report(per_run, ranks, ranks_resid, comp):
    print(f"\n{'=' * 96}")
    print("METRIC VALIDATION -- Spearman rho against the human reference")
    print("=" * 96)
    print(f"{'metric':18s} {'tier':>4} | {'rho(IoU)':>9} {'p':>8} | "
          f"{'rho(resid)':>10} | {'rho top-half':>12} | per-run rho")
    print("-" * 96)
    for name in sorted(METRICS, key=lambda n: (METRICS[n].tier, n)):
        r, rr = ranks[name], ranks_resid[name]
        pr = "  ".join(f"{run}:{v['rho']:+.2f}" for run, v in r.get("per_run", {}).items())
        flag = "" if r.get("consistent", False) else "  (sign flips)"
        print(f"{name:18s} {r['tier']:>4} | {r['pooled_rho']:+9.3f} "
              f"{r.get('pooled_p', float('nan')):8.4f} | {rr['pooled_rho']:+10.3f} | "
              f"{r['top_rho']:+12.3f} | {pr}{flag}")

    for run, c in comp.items():
        order = np.argsort(-np.asarray(c["overall"]))
        truth = [per_run[run][c["candidates"][i]]["truth_iou"] for i in order]
        print(f"\n  run {run} -- score card ranking (composite z, best first):")
        for rank, i in enumerate(order[:6], 1):
            print(f"    {rank}. {c['candidates'][i]:16s} "
                  f"composite {c['overall'][i]:+.2f}   true IoU {truth[rank-1]:.3f}")
        rho, p = stats.spearmanr(
            c["overall"], [per_run[run][n]["truth_iou"] for n in c["candidates"]])
        print(f"    composite vs truth: rho {rho:+.3f} (p {p:.4f})")


def plot_report(per_run, ranks, comp, out_path):
    names = sorted(METRICS, key=lambda n: (METRICS[n].tier, n))
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.4, 1.0])

    ax = fig.add_subplot(gs[0, :2])
    y = np.arange(len(names))
    vals = [ranks[n]["pooled_rho"] for n in names]
    tops = [ranks[n]["top_rho"] for n in names]
    colors = plt.get_cmap("viridis")(np.array([METRICS[n].tier for n in names]) / 3.0)
    ax.barh(y - 0.2, vals, height=0.4, color=colors, label="whole panel")
    ax.barh(y + 0.2, tops, height=0.4, color=colors, alpha=0.45, label="top half only")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=1); ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho vs IoU against the human mask")
    ax.set_title("does the surrogate rank masks like the truth does?\n"
                 "(faded = discrimination among the good half)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[0, 2:])
    for run, c in comp.items():
        t = [per_run[run][n]["truth_iou"] for n in c["candidates"]]
        ax.scatter(c["overall"], t, s=40, label=f"run {run}")
        for i, n in enumerate(c["candidates"]):
            ax.annotate(n, (c["overall"][i], t[i]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("composite unsupervised score card  [mean tier z]")
    ax.set_ylabel("true IoU vs human")
    ax.set_title("the score card as a whole", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for j, name in enumerate(names[:8]):
        ax = fig.add_subplot(gs[1 + j // 4, j % 4])
        for run, rows in per_run.items():
            cands = [c for c in rows if np.isfinite(rows[c].get(name, np.nan))]
            ax.scatter([rows[c][name] for c in cands],
                       [rows[c]["truth_iou"] for c in cands], s=22, label=f"{run}")
        ax.set_title(f"{name}  (tier {METRICS[name].tier}, "
                     f"rho {ranks[name]['pooled_rho']:+.2f})", fontsize=8)
        ax.set_ylabel("true IoU", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=6)

    fig.suptitle("Unsupervised mask metrics, validated against the human reference",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="*", default=list(EVAL_RUNS))
    ap.add_argument("--quick", action="store_true", help="5-candidate smoke panel")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    per_run = {}
    for run in args.runs:
        cands = build_panel(args.quick)      # fresh, so mask caches do not leak
        print(f"\n=== run {run}: {len(cands)} candidates x {len(METRICS)} metrics ===")
        per_run[run] = score_run(run, cands, seed=args.seed)

    ranks = rank_table(per_run, "truth_iou")
    ranks_resid = rank_table(per_run, "truth_residual")
    comp = composite(per_run)
    print_report(per_run, ranks, ranks_resid, comp)

    tag = "_quick" if args.quick else ""
    out_json = os.path.join(OUT_DIR, f"metric_validation{tag}.json")
    with open(out_json, "w") as f:
        json.dump({"per_run": per_run, "ranks": ranks,
                   "ranks_residual": ranks_resid, "composite": comp}, f, indent=1)
    print(f"  wrote {out_json}")
    plot_report(per_run, ranks, comp,
                os.path.join(OUT_DIR, f"metric_validation{tag}.png"))


if __name__ == "__main__":
    main()
