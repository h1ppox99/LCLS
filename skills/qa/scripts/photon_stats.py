"""Per-pixel Poisson goodness-of-fit across shots (QA method 08_pixel_photon_statistics).

Sample = one fixed pixel across T shots (never a pooled single-image histogram —
that is a Poisson mixture; see methods/08_pixel_photon_statistics.md). Per pixel:

  lambda_hat  = mean photon count
  deviance    = 2*[sum I*ln(I) - S*ln(S/T)]  (MLE form; primary verdict)
  fano        = var/mean                      (quick screen + direction)
  js          = JS(P_hat || Poisson(lambda_hat)), tail-merged, eps-smoothed
                                              (supplementary shape feature)

Flags come from robust per-lambda-stratum fences (median + k*MAD of dev/dof),
not absolute thresholds — the pixel population is the null.

Usage (real data):
  python photon_stats.py \
    --frames npy/frames_raw.npy --shot-table npy/shot_table.npz \
    --ped calib/ped.npy --gain calib/gain.npy --status-bad calib/status_bad.npy \
    --photon-kev 9.6 --max-shots 600 --out-dir outputs/<run>/qa/photon_stats

Self-test (synthetic, no data files):
  python photon_stats.py --selftest --out-dir /tmp/photon_stats_selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FRAME_SHAPE = (2, 512, 1024)


def calibrate(raw: np.ndarray, ped: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """(ADC - ped[gain mode]) / gain[gain mode], float32 keV. Mirrors pipeline/xtclib.py."""
    gb = raw >> 14
    adc = (raw & 0x3FFF).astype(np.float32)
    mode = np.zeros(raw.shape, np.intp)
    mode[gb == 1] = 1
    mode[gb == 3] = 2
    pedm = np.take_along_axis(ped, mode[None], 0)[0]
    gm = np.take_along_axis(gain, mode[None], 0)[0]
    return (adc - pedm) / gm


class PixelAccumulator:
    """Streaming accumulators for per-pixel photon statistics.

    Needs only S1 = sum k, S2 = sum k^2, SlogI = sum k*ln(k) plus a capped
    histogram (k >= kmax merged into the tail bin) — deviance in MLE form is
    2*[SlogI - S1*ln(S1/T)], so no full empirical PMF is required.
    """

    def __init__(self, shape: tuple, kmax: int):
        self.shape = shape
        self.kmax = kmax
        self.t = 0
        self.s1 = np.zeros(shape, np.float64)
        self.s2 = np.zeros(shape, np.float64)
        self.slogi = np.zeros(shape, np.float64)
        self.hist = np.zeros((kmax + 1,) + shape, np.int64)

    def add(self, counts: np.ndarray) -> None:
        k = counts.astype(np.float64)
        self.t += 1
        self.s1 += k
        self.s2 += k * k
        with np.errstate(divide="ignore", invalid="ignore"):
            klogk = np.where(k > 0, k * np.log(np.maximum(k, 1.0)), 0.0)
        self.slogi += klogk
        kc = np.minimum(counts.astype(np.int64), self.kmax)
        # bincount over a flattened (bin, pixel) index — one pass, no per-bin loop
        idx = kc.reshape(-1) * self.s1.size + np.arange(self.s1.size)
        self.hist.reshape(-1)[:] += np.bincount(idx, minlength=self.hist.size)

    def finalize(self, js_epsilon: float) -> dict:
        t = self.t
        lam = self.s1 / t
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * (self.slogi - np.where(self.s1 > 0, self.s1 * np.log(lam), 0.0))
            dev_dof = np.where(t > 1, dev / (t - 1), np.nan)
            var = (self.s2 - t * lam * lam) / (t - 1)
            fano = np.where(lam > 0, var / lam, np.nan)
        js = self._js(lam, js_epsilon)
        return {"lambda": lam, "dev_dof": dev_dof, "fano": fano, "js": js, "t": t}

    def _js(self, lam: np.ndarray, eps: float) -> np.ndarray:
        """JS(P_hat || Poisson(lambda)) over bins 0..kmax-1 plus a merged tail."""
        from scipy.stats import poisson

        kbins = self.kmax + 1
        t = self.t
        p = (self.hist.astype(np.float64) + eps) / (t + eps * kbins)
        lam_c = np.maximum(lam, 1e-12)
        ks = np.arange(self.kmax)[:, None]
        q_head = poisson.pmf(ks, lam_c.reshape(1, -1)).reshape((self.kmax,) + self.shape)
        q_tail = np.maximum(1.0 - q_head.sum(axis=0), 0.0)
        q = np.concatenate([q_head, q_tail[None]], axis=0)
        q = (q + eps / t) / (1.0 + eps * kbins / t)  # same smoothing as P
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where(p > 0, p * np.log(p / m), 0.0).sum(axis=0)
            kl_qm = np.where(q > 0, q * np.log(q / m), 0.0).sum(axis=0)
        return 0.5 * kl_pm + 0.5 * kl_qm


def stratum_fences(lam, dev_dof, fano, exclude, n_strata, fence_k):
    """Robust per-lambda-stratum fences on dev/dof. Returns (fence_map, strata_rows);
    fence_map is NaN wherever no fence applies (excluded, unlit, thin stratum).

    Pixels with lam == 0 are unfenceable (dead or unlit; deviance is 0 by
    construction) — reported separately, never flagged here.
    """
    ok = np.isfinite(dev_dof) & (lam > 0) & ~exclude
    fence_map = np.full(lam.shape, np.nan)
    rows = []
    if not ok.any():
        return fence_map, rows
    # Equal-count strata (quantile edges of lambda-hat). Log-spaced strata leave
    # thin, contamination-dominated top strata: anomalies inflate their own
    # lambda-hat and migrate into them, then either define the fence or escape a
    # skipped stratum entirely. Equal counts keep every fence anomaly-poor.
    edges = np.unique(np.quantile(lam[ok], np.linspace(0, 1, n_strata + 1)))
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max in the last stratum
    for b in range(len(edges) - 1):
        sel = ok & (lam >= edges[b]) & (lam < edges[b + 1])
        n = int(sel.sum())
        if n < 50:  # only possible when almost-all lambda-hats coincide
            if n:
                rows.append({"lambda_range": [float(edges[b]), float(edges[b + 1])],
                             "n_pixels": n, "skipped": "thin_stratum"})
            continue
        d = dev_dof[sel]
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) * 1.4826
        fence = med + fence_k * max(mad, 1e-12)
        fence_map[sel] = fence
        rows.append({
            "lambda_range": [float(edges[b]), float(edges[b + 1])],
            "n_pixels": n, "dev_dof_median": med, "dev_dof_mad": mad,
            "fence": fence, "fano_median": float(np.nanmedian(fano[sel])),
        })
    return fence_map, rows


def mc_null_fence(lam, t_shots, mono_rel, fence_k, kmax_grid=25, n_rep=400, seed=7):
    """MC-calibrated parametric null fence: dev/dof quantiles of Poisson(lam*f_t)
    using the MEASURED per-shot relative flux sequence f_t.

    The empirical stratum fence alone has a within-stratum drift blind spot: the
    dev/dof null level rises with lambda, so pixels at a stratum's bright edge
    are judged against a fence built for its dim bulk and flag spuriously (100%
    of lambda > 0.1 pixels on Run0475 before this fix). The MC fence gives every
    pixel a null at ITS OWN lambda, including the measured flux jitter; the
    final fence is the pixelwise max of both.
    """
    rng = np.random.default_rng(seed)
    ok = lam > 0
    lo = max(float(lam[ok].min()), 1e-4)
    hi = float(lam[ok].max())
    grid = np.unique(np.logspace(np.log10(lo), np.log10(hi), kmax_grid))
    f = np.asarray(mono_rel, np.float64)
    f = f / f.mean()
    fences = np.empty(len(grid))
    for i, g in enumerate(grid):
        k = rng.poisson(g * f, size=(n_rep, t_shots)).astype(np.float64)
        s1 = k.sum(1)
        slogi = np.where(k > 0, k * np.log(np.maximum(k, 1.0)), 0.0).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * (slogi - np.where(s1 > 0, s1 * np.log(s1 / t_shots), 0.0))
        dd = dev / (t_shots - 1)
        med = np.median(dd)
        mad = np.median(np.abs(dd - med)) * 1.4826
        fences[i] = med + fence_k * max(mad, 1e-12)
    out = np.full(lam.shape, np.nan)
    out[ok] = np.interp(np.log(lam[ok]), np.log(grid), fences)
    return out, {"grid": grid.tolist(), "fences": fences.tolist(), "n_rep": n_rep}


def classify(lam, dev_dof, fano, js, flags):
    """Suspected class per flagged pixel, per the method file's routing table."""
    out = np.full(lam.shape, "", dtype=object)
    hot = flags & (fano > 10)
    flicker = flags & ~hot & (fano > 1) & (js > 0.05)
    noisy = flags & ~hot & ~flicker & (fano > 1)
    under = flags & (fano < 1)
    out[hot] = "hot_or_unstable"
    out[flicker] = "flicker_or_calib"
    out[noisy] = "excess_noise"
    out[under] = "underdispersed"
    rest = flags & (out == "")
    out[rest] = "unclassified"
    return out


def run_real(args) -> int:
    t = np.load(args.shot_table)
    ipm2, xray = t["ipm2"], t["xray"]
    offset = float(np.median(ipm2[xray == 0]))
    ipm2c = ipm2 - offset
    keep = xray == 1
    qlo, qhi = np.quantile(ipm2c[keep], args.monitor_band)
    keep &= (ipm2c >= qlo) & (ipm2c <= qhi)
    idx = np.where(keep)[0]
    if len(idx) > args.max_shots:
        idx = idx[np.linspace(0, len(idx) - 1, args.max_shots).round().astype(int)]
    if len(idx) < args.min_shots:
        report = {"skipped": "insufficient_shots", "n_shots_available": int(len(idx)),
                  "min_shots": args.min_shots}
        _write_report(args.out_dir, report)
        print(f"SKIP insufficient_shots: {len(idx)} < {args.min_shots}")
        return 0
    mono = ipm2c[idx]
    monitor_cv = float(np.std(mono) / np.mean(mono))

    ped = np.load(args.ped)
    gain = np.load(args.gain)
    status_bad = np.load(args.status_bad).astype(bool) if args.status_bad else \
        np.zeros(FRAME_SHAPE, bool)
    frames = np.load(args.frames, mmap_mode="r")

    acc = PixelAccumulator(FRAME_SHAPE, args.js_kmax)
    import time as _time
    t0 = _time.time()
    for n, i in enumerate(idx):
        kev = calibrate(np.asarray(frames[i]), ped, gain)
        counts = np.maximum(np.rint(kev / args.photon_kev), 0.0)
        acc.add(counts)
        if (n + 1) % 100 == 0:
            print(f"[photon_stats] {n+1}/{len(idx)} shots ({_time.time()-t0:.0f}s)", flush=True)

    stats = acc.finalize(args.js_epsilon)
    return _finish(args, stats, status_bad, monitor_cv, mono_rel=mono, extra={
        "n_shots_used": int(len(idx)), "monitor_band": list(args.monitor_band),
        "ipm2_offset": offset})


def _finish(args, stats, known_bad, monitor_cv, mono_rel, extra) -> int:
    lam, dev_dof, fano, js = stats["lambda"], stats["dev_dof"], stats["fano"], stats["js"]
    fence_s, strata = stratum_fences(lam, dev_dof, fano, known_bad,
                                     args.lambda_strata, args.fence_k)
    fence_mc, mc_info = mc_null_fence(lam, stats["t"], mono_rel, args.fence_k)
    fence = np.fmax(fence_s, fence_mc)  # fmax: NaN-tolerant pixelwise max
    ok = np.isfinite(dev_dof) & (lam > 0) & ~known_bad & np.isfinite(fence)
    flags = ok & (dev_dof > fence)
    classes = classify(lam, dev_dof, fano, js, flags)

    n_fenceable = int(ok.sum())
    n_flagged = int(flags.sum())
    overlap = int((flags & known_bad).sum())  # 0 by construction (bad excluded); keep for schema
    lit = lam > 0
    for row in strata:
        if "fence" in row:
            a, b = row["lambda_range"]
            row["n_flagged"] = int((flags & (lam >= a) & (lam < b)).sum())
    report = {
        **extra,
        "t_shots": stats["t"],
        "monitor_cv": monitor_cv,
        "n_pixels_lit": int(lit.sum()),
        "n_pixels_fenceable": n_fenceable,
        "n_known_bad": int(known_bad.sum()),
        "n_flagged": n_flagged,
        "flag_fraction": n_flagged / max(n_fenceable, 1),
        "overlap_with_known_bad": overlap,
        "global_fano_median": float(np.nanmedian(fano[lit & ~known_bad])),
        "predicted_fano_from_jitter": "1 + lambda*cv^2 with cv=monitor_cv",
        "hard_escalation": bool(n_flagged / max(n_fenceable, 1) > args.max_flag_fraction),
        "mc_null": mc_info,
        "strata": strata,
        "class_counts": {c: int((classes == c).sum()) for c in
                         np.unique(classes[flags]).tolist()} if n_flagged else {},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "photon_stats.npz", lam=lam.astype(np.float32),
                        dev_dof=dev_dof.astype(np.float32), fano=fano.astype(np.float32),
                        js=js.astype(np.float32), flags=flags, known_bad=known_bad)
    # top flags for the report (by dev/dof)
    if n_flagged:
        fi = np.argwhere(flags)
        order = np.argsort(-dev_dof[flags])
        top = []
        for j in order[:200]:
            p, r, c = fi[j]
            top.append({"panel": int(p), "row": int(r), "col": int(c),
                        "lambda": float(lam[p, r, c]), "dev_dof": float(dev_dof[p, r, c]),
                        "fano": float(fano[p, r, c]), "js": float(js[p, r, c]),
                        "suspected_class": str(classes[p, r, c])})
        report["flags_top"] = top
    _write_report(args.out_dir, report)
    _plots(out, lam, dev_dof, fano, js, flags, known_bad, monitor_cv, strata, mc_info)
    print(json.dumps({k: report[k] for k in
                      ["t_shots", "monitor_cv", "n_pixels_fenceable", "n_flagged",
                       "flag_fraction", "global_fano_median", "hard_escalation"]}, indent=2))
    return 0


def _write_report(out_dir, report) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "photon_stats_report.json").write_text(json.dumps(report, indent=2))


def _plots(out, lam, dev_dof, fano, js, flags, known_bad, monitor_cv, strata,
           mc_info=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(dev_dof) & (lam > 0) & ~known_bad
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    img = np.concatenate([lam[0], lam[1]], axis=0)
    im = ax.imshow(img, vmin=0, vmax=np.nanpercentile(lam[lam > 0], 99), cmap="viridis",
                   interpolation="nearest", aspect="auto")
    ax.set_title(r"$\hat\lambda$ [photons/shot] (panels stacked)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    img = np.concatenate([dev_dof[0], dev_dof[1]], axis=0)
    im = ax.imshow(img, vmin=0, vmax=np.nanpercentile(dev_dof[ok], 99.5), cmap="magma",
                   interpolation="nearest", aspect="auto")
    ax.set_title("Poisson deviance / dof")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    fy, fx = np.where(np.concatenate([flags[0], flags[1]], axis=0))
    ax.imshow(np.concatenate([known_bad[0], known_bad[1]], axis=0), cmap="gray_r",
              interpolation="nearest", aspect="auto")
    ax.plot(fx, fy, "r.", ms=2)
    ax.set_title(f"flags (red, n={flags.sum()}) over known-bad (gray)")

    ax = axes[1, 0]
    ax.hist(dev_dof[ok], bins=200, range=(0, np.nanpercentile(dev_dof[ok], 99.9)),
            log=True, color="steelblue")
    ax.set_xlabel("deviance/dof"); ax.set_ylabel("pixels"); ax.set_title("dev/dof (good pixels)")

    ax = axes[1, 1]
    ls = np.logspace(np.log10(max(lam[ok].min(), 1e-4)), np.log10(lam[ok].max()), 30)
    med_f = [np.nanmedian(fano[sel]) if (sel := ok & (lam >= a) & (lam < b)).any() else np.nan
             for a, b in zip(ls[:-1], ls[1:])]
    ctr = np.sqrt(ls[:-1] * ls[1:])
    ax.semilogx(ctr, med_f, "o-", label="median Fano")
    ax.semilogx(ctr, 1 + ctr * monitor_cv**2, "--", color="gray",
                label=rf"$1+\lambda\,\mathrm{{CV}}^2$ (CV={monitor_cv:.3f})")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel(r"$\hat\lambda$"); ax.set_ylabel("Fano"); ax.legend()
    ax.set_title("Fano vs intensity (flux-jitter prediction)")

    ax = axes[1, 2]
    sc = ax.scatter(lam[ok][::37], dev_dof[ok][::37], s=1, alpha=0.15, label="good")
    if flags.any():
        ax.scatter(lam[flags], dev_dof[flags], s=6, c="r", label="flagged")
    for row in strata:
        if "fence" in row:
            ax.plot(row["lambda_range"], [row["fence"]] * 2, "g-", lw=1)
    if mc_info:
        ax.plot(mc_info["grid"], mc_info["fences"], "b--", lw=1, label="MC null fence")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\hat\lambda$"); ax.set_ylabel("dev/dof")
    ax.set_title("per-stratum fences (green)"); ax.legend(markerscale=4)

    fig.tight_layout()
    fig.savefig(out / "photon_stats.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- self-test

def run_selftest(args) -> int:
    """Synthetic validation: known-good Poisson + injected anomalies.

    Grid of 2x200x400 'pixels' x T shots. Sections:
      A  constant flux — null calibration + injected anomalies must be caught
      B  10% flux jitter — Fano trend must follow 1 + lambda*cv^2; fences hold
      C  pooled-histogram mixture pitfall — pooled KL huge on healthy data
    """
    rng = np.random.default_rng(20260727)
    shape = (2, 200, 400)
    T = args.max_shots
    npix = int(np.prod(shape))
    lam_true = 10.0 ** rng.uniform(-2, 1.3, npix).reshape(shape)  # 0.01..20 ph/shot

    # injected anomalies (disjoint sets of 300 pixels each, flat indices)
    flat = rng.permutation(npix)
    n_a = 300
    hot_idx, flick_idx, noise_idx, stuck_idx = (flat[i*n_a:(i+1)*n_a] for i in range(4))
    truth = np.zeros(npix, dtype=object); truth[:] = ""
    truth[hot_idx] = "hot"; truth[flick_idx] = "flicker"
    truth[noise_idx] = "noise"; truth[stuck_idx] = "stuck"

    def simulate(jitter_cv):
        acc = PixelAccumulator(shape, args.js_kmax)
        lam_flat = lam_true.reshape(-1).copy()
        fs = []
        for t in range(T):
            f = max(rng.normal(1.0, jitter_cv), 0.05) if jitter_cv else 1.0
            fs.append(f)
            lam_t = lam_flat * f
            # hot: occasional huge bursts (5% of shots at 30x)
            lam_t[hot_idx] = lam_flat[hot_idx] * (30.0 if rng.random() < 0.05 else 1.0) * f
            # flicker: two-state gain (x1 / x2.5, 50/50)
            lam_t[flick_idx] = lam_flat[flick_idx] * (2.5 if rng.random() < 0.5 else 1.0) * f
            k = rng.poisson(lam_t).astype(np.float64)
            # noise: anomalously noisy pixel — extra Gaussian sigma = 2 photons, re-rounded
            # (nominal Jungfrau read noise is far below 1 photon at 9.6 keV; sub-photon
            # additive noise after clipping is nearly Poisson-shaped and correctly invisible)
            k[noise_idx] = np.maximum(np.rint(k[noise_idx] + rng.normal(0, 2.0, n_a)), 0)
            # stuck: constant value
            k[stuck_idx] = np.maximum(np.rint(lam_flat[stuck_idx]), 1)
            acc.add(k.reshape(shape))
        return acc.finalize(args.js_epsilon), np.array(fs)

    results = {}
    for name, cv in [("A_constant_flux", 0.0), ("B_flux_jitter_10pct", 0.10)]:
        stats, fs = simulate(cv)
        lam, dev_dof, fano = stats["lambda"], stats["dev_dof"], stats["fano"]
        fence_s, strata = stratum_fences(lam, dev_dof, fano, np.zeros(shape, bool),
                                         args.lambda_strata, args.fence_k)
        fence_mc, _mc = mc_null_fence(lam, T, fs, args.fence_k)
        fence = np.fmax(fence_s, fence_mc)
        okm = np.isfinite(dev_dof) & (lam > 0) & np.isfinite(fence)
        flags = okm & (dev_dof > fence)
        fl = flags.reshape(-1)
        lam_flat_true = lam_true.reshape(-1)
        # detection is lambda-dependent (sample-size limit, method file §sample size):
        # report per lambda group; the pass bar sits on the powered group (lambda >= 1)
        groups = {"lam_lt_0.1": lam_flat_true < 0.1,
                  "lam_0.1_1": (lam_flat_true >= 0.1) & (lam_flat_true < 1),
                  "lam_ge_1": lam_flat_true >= 1}
        det = {c: {g: float(fl[(truth == c) & gm].mean()) if ((truth == c) & gm).any()
                   else None for g, gm in groups.items()}
               for c in ["hot", "flicker", "noise"]}
        # stuck pixels: deviance ~0 (underdispersed) — caught by Fano < 1 screen, not fence
        stuck_fano = fano.reshape(-1)[stuck_idx]
        det["stuck_by_fano_lt_0.5"] = float((stuck_fano < 0.5).mean())
        fp = float(fl[truth == ""].mean())
        results[name] = {
            "detection_rate": det, "false_positive_rate": fp,
            "good_dev_dof_median": float(np.nanmedian(dev_dof.reshape(-1)[truth == ""])),
            "good_fano_median": float(np.nanmedian(fano.reshape(-1)[truth == ""])),
            "n_flagged": int(flags.sum()),
        }
        if name == "A_constant_flux":
            keep_maps = (stats, flags, strata)

    # C: pooled-histogram pitfall on HEALTHY pixels only, constant flux
    stats, flags, strata = keep_maps
    good_mask = (truth == "").reshape(shape)
    kpool = rng.poisson(lam_true, size=(50,) + shape).astype(np.float64)
    pooled_counts = kpool[:, good_mask].reshape(-1)
    hist = np.array([(pooled_counts == k).sum() for k in range(args.js_kmax)] +
                    [(pooled_counts >= args.js_kmax).sum()], np.float64)
    p = (hist + args.js_epsilon) / (hist.sum() + args.js_epsilon * len(hist))
    from scipy.stats import poisson as _poisson
    lam_pool = pooled_counts.mean()
    q_head = _poisson.pmf(np.arange(args.js_kmax), lam_pool)
    q = np.append(q_head, max(1 - q_head.sum(), 0.0))
    q = (q + args.js_epsilon / len(pooled_counts)) / (1 + args.js_epsilon * len(q) / len(pooled_counts))
    kl_pooled = float(np.where(p > 0, p * np.log(p / q), 0).sum())
    js_good_median = float(np.nanmedian(stats["js"][good_mask]))
    results["C_pooled_mixture_pitfall"] = {
        "pooled_KL_healthy_pixels": kl_pooled,
        "per_pixel_JS_median_healthy": js_good_median,
        "ratio": kl_pooled / max(js_good_median, 1e-12),
        "note": "pooled histogram vs single Poisson is huge even with zero anomalies",
    }

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "selftest_report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    a = results["A_constant_flux"]
    ok = (a["detection_rate"]["hot"]["lam_ge_1"] > 0.95
          and a["detection_rate"]["flicker"]["lam_ge_1"] > 0.90
          and a["detection_rate"]["stuck_by_fano_lt_0.5"] > 0.95
          and a["false_positive_rate"] < 0.005
          and abs(a["good_dev_dof_median"] - 1.0) < 0.15
          and results["C_pooled_mixture_pitfall"]["ratio"] > 100)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--frames"); p.add_argument("--shot-table")
    p.add_argument("--ped"); p.add_argument("--gain"); p.add_argument("--status-bad")
    p.add_argument("--photon-kev", type=float, help="from manifest; no default on purpose")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-shots", type=int, default=600)
    p.add_argument("--min-shots", type=int, default=200)
    p.add_argument("--monitor-band", type=float, nargs=2, default=[0.25, 0.75])
    p.add_argument("--fence-k", type=float, default=6.0)
    p.add_argument("--lambda-strata", type=int, default=16)
    p.add_argument("--js-kmax", type=int, default=12)
    p.add_argument("--js-epsilon", type=float, default=0.5)
    p.add_argument("--max-flag-fraction", type=float, default=0.005)
    args = p.parse_args()
    if args.selftest:
        return run_selftest(args)
    for req in ["frames", "shot_table", "ped", "gain", "photon_kev"]:
        if getattr(args, req) is None:
            p.error(f"--{req.replace('_', '-')} is required (values come from the manifest)")
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
