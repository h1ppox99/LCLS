#!/usr/bin/env python3
"""
studies/frangi_tuning.py -- justify the Frangi scales `s` and `beta`.

Locked production choice: sigmas=(1,2,3), beta=0.5, gamma=None (the FrangiParams
defaults). This study shows WHY, unsupervised:

  * The only usable configs are MULTISCALE sets spanning the ridge widths
    (~1-3 px). Single scales / narrow pairs respond too sparsely to form a ridge
    mode, so the auto-threshold (threshold_minimum on log10 response) collapses
    into the noise floor (k ~ 0) and masks nothing.
  * A config is accepted iff its auto-threshold is NON-DEGENERATE on every run
    (k in [1e-3, 1e-1], i.e. above the floor) AND run-stable (low k spread).
    That uniquely selects {1,2,3} / {1,2,3,4,5}; {1,2,3} is the most stable.
  * beta barely moves anything -- 0.5 is the scale-invariant default.

gamma stays None (keeps k run-agnostic). `s` values are capped at 5 px. IoU vs
the human mask is a cross-check only (no ground truth in production).

Outputs (outputs/figures/frangi_mask/):
    frangi_tuning_autok.png     log10(auto-k) over (sigma-set x beta)
    frangi_tuning_sweep.png     log-histograms across s (beta=0.5, run 475)
    frangi_tuning_bestmask.png  best-config ridge overlay on the sum image

Run:  python -m automask.studies.frangi_tuning
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum

from automask.evaluation import load_sample, EVAL_RUNS
from automask.masking import production_pipeline
from automask.dataset import score
from automask.regularization.frangi import frangi_ridges

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "outputs", "figures", "frangi_mask")
os.makedirs(OUTDIR, exist_ok=True)

FLOOR = 1e-9

# scales `s`: single-scale ladder + a few multi-scale sets, all values <= 5 px.
SIGMA_SETS = [
    (1.0,), (2.0,), (3.0,), (4.0,), (5.0,),
    (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
    (1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (3.0, 4.0, 5.0),
    (1.0, 2.0, 3.0, 4.0, 5.0),
]
BETAS = [0.3, 0.5, 0.75, 1.0]


def sset_label(s):
    return "-".join(str(int(x)) for x in s)


def current_mask(sample):
    """geometry+calib floor UNION the production variance detector pick."""
    pipe = production_pipeline("union")
    floor = pipe.floor(sample)
    var = next(d for d in pipe.detectors if d.stat == "variance")
    return floor | var.pick(sample)


def prep_runs():
    ctx = {}
    for run in EVAL_RUNS:
        s = load_sample(run)
        cur = current_mask(s)
        ctx[run] = dict(sample=s, cur=cur, domain=s.real & ~cur)
    return ctx


def eval_config(ctx, sigmas, beta):
    """Per-run dict of auto-k (threshold_minimum) and IoU for one (s, beta)."""
    out = {}
    for run, c in ctx.items():
        resp = frangi_ridges(c["sample"].sumimg, sigmas=sigmas, beta=beta,
                             gamma=None, black_ridges=False)
        v = resp[c["domain"]]
        v = v[v > FLOOR]
        logv = np.log10(v)
        try:
            k = float(10.0 ** threshold_minimum(logv, nbins=256))
        except RuntimeError:      # no bimodal valley -> degenerate
            k = float("nan")
        s = c["sample"]
        iou = (score(c["cur"] | ((resp > k) & s.real), s.human)["iou"]
               if np.isfinite(k) else float("nan"))
        out[run] = dict(k=k, iou=iou, resp=resp)
    return out


def main():
    ctx = prep_runs()
    runs = list(EVAL_RUNS)

    k_grid = np.full((len(SIGMA_SETS), len(BETAS)), np.nan)   # mean auto-k
    records = {}
    print(f"{'sigmas':14s} {'beta':>5s} | "
          + " ".join(f"{'k.'+str(r):>8s} {'IoU.'+str(r):>7s}" for r in runs))
    print("-" * 70)
    for si, sig in enumerate(SIGMA_SETS):
        for bi, beta in enumerate(BETAS):
            res = eval_config(ctx, sig, beta)
            records[(si, bi)] = res
            k_grid[si, bi] = float(np.nanmean([res[r]["k"] for r in runs]))
            row = f"{sset_label(sig):14s} {beta:5.2f} | "
            for r in runs:
                row += f"{res[r]['k']:8.4f} {res[r]['iou']:7.3f} "
            print(row)

    # ---- unsupervised & grounded selection -----------------------------------
    # auto-threshold must be non-degenerate (above the floor) on every run and
    # run-stable; rank survivors by k spread, then IoU cross-check.
    LO, HI = 1e-3, 1e-1
    cand = []
    for (si, bi), res in records.items():
        ks = np.array([res[r]["k"] for r in runs])
        if np.all(np.isfinite(ks)) and np.all((ks > LO) & (ks < HI)):
            cand.append(((si, bi), float(np.std(ks)),
                         float(np.mean([res[r]["iou"] for r in runs]))))
    cand.sort(key=lambda t: (t[1], -t[2]))
    best = cand[0][0]
    best_sig, best_beta = SIGMA_SETS[best[0]], BETAS[best[1]]
    print(f"\nnon-degenerate & run-stable configs (k in [{LO},{HI}]):")
    for (si, bi), kstd, iou in cand:
        print(f"    s={sset_label(SIGMA_SETS[si]):8s} beta={BETAS[bi]:.2f}  "
              f"k_std={kstd:.2e}  meanIoU={iou:.3f}")
    print(f"\nBEST: sigmas={sset_label(best_sig)} beta={best_beta}")
    for r in runs:
        rr = records[best][r]
        print(f"  run {r}: auto-k {rr['k']:.4f}  IoU {rr['iou']:.3f}")

    # ---- figure 1: log10(auto-k) heatmap -------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(np.log10(k_grid), aspect="auto", cmap="magma", origin="upper")
    ax.set_xticks(range(len(BETAS))); ax.set_xticklabels([str(b) for b in BETAS])
    ax.set_yticks(range(len(SIGMA_SETS)))
    ax.set_yticklabels([sset_label(s) for s in SIGMA_SETS])
    ax.set_xlabel("beta"); ax.set_ylabel("sigmas s (px)")
    ax.set_title("log10(auto-k)  (mean of runs 389/475)\n"
                 "bright band ~ -1.9 = non-degenerate ridge cut (usable); "
                 "dark = collapses into floor")
    for si in range(len(SIGMA_SETS)):
        for bi in range(len(BETAS)):
            ax.text(bi, si, f"{k_grid[si,bi]:.4f}", ha="center", va="center",
                    color="w", fontsize=7)
    ax.scatter([best[1]], [best[0]], s=280, facecolors="none",
               edgecolors="lime", linewidths=2.5)
    fig.colorbar(im, ax=ax, label="log10(auto-k)")
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, "frangi_tuning_autok.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- figure 2: log-hist across s at beta=0.5, run 475 --------------------
    bi05 = BETAS.index(0.5)
    run_show = 475 if 475 in runs else runs[0]
    n = len(SIGMA_SETS); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for si, sig in enumerate(SIGMA_SETS):
        ax = axes[si]
        res = records[(si, bi05)][run_show]
        v = res["resp"][ctx[run_show]["domain"]]; v = v[v > FLOOR]
        ax.hist(np.log10(v), bins=200, color="#3b6ea5", alpha=0.85)
        if np.isfinite(res["k"]):
            ax.axvline(np.log10(res["k"]), color="crimson", lw=1.6)
        ax.set_title(f"s={sset_label(sig)}  k={res['k']:.4f}", fontsize=9)
        ax.set_xlim(-9, 0); ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"log10(Frangi response) vs scale s  (beta=0.5, run {run_show}) "
                 f"-- red = threshold_minimum cut", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = os.path.join(OUTDIR, "frangi_tuning_sweep.png")
    fig.savefig(p2, dpi=130); plt.close(fig)

    # ---- figure 3: best-config ridge overlay on the sum image ----------------
    fig, axes = plt.subplots(1, len(runs), figsize=(7.5 * len(runs), 7.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        c = ctx[r]; s = c["sample"]; res = records[best][r]
        ridges = (res["resp"] > res["k"]) & s.real & ~c["cur"]
        base = np.arcsinh(s.sumimg / (np.nanmedian(np.abs(s.sumimg[s.real])) + 1e-9))
        vlo, vhi = np.nanpercentile(base[s.real], [2, 99])
        ax.imshow(base.T, cmap="gray", vmin=vlo, vmax=vhi, origin="lower")
        red = np.zeros((*s.sumimg.shape, 4)); red[ridges] = (1, 0, 0, 1)
        ax.imshow(np.transpose(red, (1, 0, 2)), origin="lower")
        ax.set_title(f"run {r}: +{100*ridges.mean():.3f}% ridge px  "
                     f"(s={sset_label(best_sig)}, beta={best_beta}, k={res['k']:.4f})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Best-config Frangi ridges (red) added beyond geometry+variance mask",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = os.path.join(OUTDIR, "frangi_tuning_bestmask.png")
    fig.savefig(p3, dpi=130); plt.close(fig)

    print(f"\nsaved:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
