"""Across-shot statistical pixel-mask layers (masking methods 07 and 08).

Consumes the per-pixel maps produced by qa/scripts/photon_stats.py
(photon_stats.npz: lam, fano, js, known_bad) and applies the mask-side fences:

  --stat fano   07: dispersion screen. Upper flag = F over the pixelwise max of
                (per-lambda-stratum median + k*MAD) and an MC fence simulated
                with the measured monitor sequence f_t. Lower flag = F below
                stuck_fano (absolute backstop; deviance is ~0 for stuck pixels
                so only Fano catches them).
  --stat js     08: shape screen. Same two-fence structure applied to
                JS(P_hat || Poisson(lambda_hat)) (kmax=12 head bins + merged
                tail, eps=0.5 smoothing — photon_stats.py's convention).

Flags are evidence with a class label; the layer written here excludes classes
routed elsewhere (beam-modulated pixels are NOT masked by default).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson

KHEAD = 12
EPS = 0.5


def robust_fence(x, k):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, mad, med + k * max(mad, 1e-12)


def stratum_fences(lam, stat, ok, n_strata, fence_k):
    fence = np.full(lam.shape, np.nan)
    edges = np.unique(np.quantile(lam[ok], np.linspace(0, 1, n_strata + 1)))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = ok & (lam >= a) & (lam < b)
        if sel.sum() < 50:
            continue
        med, mad, f = robust_fence(stat[sel], fence_k)
        fence[sel] = f
        rows.append({"lambda_range": [float(a), float(b)], "n": int(sel.sum()),
                     "median": med, "mad": mad, "fence": f})
    return fence, rows


def js_of_sims(sim, T):
    """JS(P_hat || Poisson(lam_hat)) per row of sim, photon_stats convention.
    Vectorized across replicas (histogram scatter-add + batched pmf)."""
    n = len(sim)
    h = np.zeros((KHEAD + 1, n), np.int64)
    np.add.at(h, (np.minimum(sim, KHEAD).reshape(-1),
                  np.repeat(np.arange(n), T)), 1)
    p = (h.astype(np.float64) + EPS) / (T + EPS * (KHEAD + 1))
    lam_r = np.maximum(sim.mean(axis=1), 1e-12)
    qh = poisson.pmf(np.arange(KHEAD)[:, None], lam_r[None, :])
    q = np.vstack([qh, np.maximum(1 - qh.sum(axis=0), 0.0)[None]])
    q = (q + EPS / T) / (1.0 + EPS * (KHEAD + 1) / T)
    m = 0.5 * (p + q)
    return (0.5 * np.where(p > 0, p * np.log(p / m), 0).sum(axis=0)
            + 0.5 * np.where(q > 0, q * np.log(q / m), 0).sum(axis=0))


def mc_fence(lam, f, T, stat_name, alpha, n_grid=25, n_rep=20000, seed=7,
             chunk=5000):
    """Per-lambda MC null QUANTILE fence at 1-alpha, measured f_t sequence.

    Quantile, not median+k*MAD: Fano/JS live on a discrete lattice at low
    T*lambda (F steps of ~2/S), where the MAD collapses and a k*MAD fence
    lands between lattice points, mass-flagging ordinary pixels. The MC
    replicas reproduce the exact lattice, so their empirical quantile is
    correct at any occupancy.
    """
    rng = np.random.default_rng(seed)
    ok = lam > 0
    grid = np.unique(np.logspace(np.log10(max(lam[ok].min(), 1e-4)),
                                 np.log10(lam[ok].max()), n_grid))
    fences = np.empty(len(grid))
    for i, g in enumerate(grid):
        stats = []
        for lo in range(0, n_rep, chunk):
            nr = min(chunk, n_rep - lo)
            sim = rng.poisson(g * f[None, :], size=(nr, T))
            if stat_name == "fano":
                lam_r = np.maximum(sim.mean(axis=1), 1e-12)
                stats.append(sim.var(axis=1, ddof=1) / lam_r)
            else:
                stats.append(js_of_sims(sim, T))
        fences[i] = np.quantile(np.concatenate(stats), 1.0 - alpha)
    out = np.full(lam.shape, np.nan)
    out[ok] = np.interp(np.log(lam[ok]), np.log(grid), fences)
    return out, {"grid": grid.tolist(), "fences": fences.tolist(),
                 "n_rep": n_rep, "alpha": alpha}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stat", choices=["fano", "js"], required=True)
    p.add_argument("--photon-stats", required=True,
                   help="photon_stats.npz from qa/scripts/photon_stats.py")
    p.add_argument("--shot-table", required=True)
    p.add_argument("--monitor-band", type=float, nargs=2, default=[0.45, 0.55],
                   help="must match the photon_stats run")
    p.add_argument("--max-shots", type=int, default=600)
    p.add_argument("--fence-k", type=float, default=6.0)
    p.add_argument("--lambda-strata", type=int, default=16)
    p.add_argument("--stuck-fano", type=float, default=0.5)
    p.add_argument("--min-photons", type=float, default=30.0,
                   help="fence only pixels with T*lambda >= this; below it the "
                        "count lattice is discrete, the stratum MAD collapses "
                        "to 0 and a median+k*MAD fence degenerates")
    p.add_argument("--mc-alpha", type=float, default=1e-4,
                   help="per-pixel false-positive target of the MC quantile "
                        "fence (expected chance flags = alpha * n_fenceable)")
    p.add_argument("--mc-reps", type=int, default=20000)
    p.add_argument("--beam-radius-px", type=float, default=60.0,
                   help="assembled radius inside which overdispersion is "
                        "classified beam_modulated and routed, not masked")
    p.add_argument("--ix")
    p.add_argument("--iy")
    p.add_argument("--image-center", type=float, nargs=2,
                   help="assembled ROW COL; otherwise load <out-dir>/image_center.json")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    m = np.load(args.photon_stats)
    lam, known_bad = m["lam"].astype(np.float64), m["known_bad"]
    stat = m[args.stat].astype(np.float64)
    ok = (lam > 0) & ~known_bad & np.isfinite(stat)

    # measured monitor sequence of the same selection as the photon_stats run
    t = np.load(args.shot_table)
    ipm2, xray = t["ipm2"], t["xray"]
    ipm2c = ipm2 - float(np.median(ipm2[xray == 0]))
    keep = xray == 1
    qlo, qhi = np.quantile(ipm2c[keep], args.monitor_band)
    keep &= (ipm2c >= qlo) & (ipm2c <= qhi)
    idx = np.where(keep)[0]
    if len(idx) > args.max_shots:
        idx = idx[np.linspace(0, len(idx) - 1, args.max_shots).round().astype(int)]
    f = ipm2c[idx].astype(np.float64)
    f /= f.mean()
    T = len(idx)

    n_below_gate = int((ok & (lam * T < args.min_photons)).sum())
    ok &= lam * T >= args.min_photons

    # stratum med+MAD kept as a reported DIAGNOSTIC only (lattice-unsafe as a
    # flag criterion — see mc_fence docstring); the flag comes from the MC
    # quantile fence alone.
    _, strata = stratum_fences(lam, stat, ok, args.lambda_strata, args.fence_k)
    fence, mc = mc_fence(np.where(ok, lam, 0.0), f, T, args.stat,
                         args.mc_alpha, n_rep=args.mc_reps)
    upper = ok & (stat > fence)
    lower = ok & (stat < args.stuck_fano) if args.stat == "fano" \
        else np.zeros_like(ok)

    # classify beam-vicinity overdispersion -> routed, not masked
    beam_modulated = np.zeros_like(ok)
    if args.ix and args.iy:
        center = args.image_center
        if center is None:
            artifact = Path(args.out_dir) / "image_center.json"
            if not artifact.exists():
                p.error("--image-center or <out-dir>/image_center.json is required with --ix/--iy")
            center_obj = json.loads(artifact.read_text()).get("center") or {}
            center = [float(center_obj["row"]), float(center_obj["col"])]
        ix = np.load(args.ix).reshape(lam.shape).astype(np.float64)
        iy = np.load(args.iy).reshape(lam.shape).astype(np.float64)
        rad = np.hypot(ix - center[0], iy - center[1])
        beam_modulated = upper & (rad < args.beam_radius_px)

    mask_layer = (upper & ~beam_modulated) | lower
    classes = {"overdispersed_masked": int((upper & ~beam_modulated).sum()),
               "beam_modulated_routed": int(beam_modulated.sum()),
               "stuck_masked": int(lower.sum())}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"mask_{args.stat}.npy", mask_layer)
    report = {"stat": args.stat, "t_shots": T,
              "monitor_cv": float(np.std(f)), "fence_k": args.fence_k,
              "min_photons": args.min_photons, "n_below_gate": n_below_gate,
              "n_fenceable": int(ok.sum()), "n_mask": int(mask_layer.sum()),
              "mask_fraction": float(mask_layer.sum() / max(ok.sum(), 1)),
              "classes": classes, "stat_median": float(np.median(stat[ok])),
              "strata": strata, "mc": mc}
    (out / f"mask_{args.stat}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ["stat", "t_shots", "n_fenceable", "n_mask",
                       "mask_fraction", "classes", "stat_median"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
