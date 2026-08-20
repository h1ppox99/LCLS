"""Per-shot Poisson conformity screen (selection method 02_shot_poisson_conformity).

The transpose of QA method 08 (pixel photon statistics): there the sample is one
pixel across shots; here it is one SHOT across pixels, judged against the same
conditional null  k_i(t) ~ Poisson(lambda_i * f_t)  with the measured per-shot
monitor flux f_t. A shot is dropped for *disagreeing with the null*, never for
being bright — bright-but-conforming shots are exactly the ones worth keeping,
which is what replaces a fixed bright-tail percentile cut (selection 01b) in its
protective role.

Per shot t (healthy pixels only: ~status_bad and lambda_hat > 0):

  S_t      = total photons;  M_t = f_t * Lambda  (Lambda = sum of lambda_hat)
  r_t      = S_t/M_t - 1                relative excess vs monitor prediction
  D_t/N    = mean per-pixel Poisson deviance vs the SELF-normalized expectation
             mu_i = lambda_hat_i * (S_t/Lambda) — tests spatial SHAPE only,
             deliberately blind to the flux mismatch that r_t already measures
  n_spike  = pixels with k above the per-pixel Poisson tail bound at the shot's
             own brightness (zingers / burst pixels)

The dominant residual of r_t on a full flux range is MULTIPLICATIVE (monitor
nonlinearity as a function of flux level, plus slow pointing/spectral drift in
time) — a Poisson-scaled z = (S-M)/sqrt(M) would re-flag ordinary multiplicative
scatter on bright shots simply because it grows like sqrt(M). So r_t and D_t are
each detrended twice — a rolling median along the FLUX axis (conditional
expectation vs monitor level), then along TIME (drift) — and r is standardized
by a combined error model  sigma_t = sqrt(sigma_mult^2 + 1/M_t)  before fencing
at fence_k * MAD across the shot population. Flags mean "outlier against peers
at the same flux and epoch", never "bright".

Usage:
  python shot_poisson_conformity.py \
    --frames npy/frames_raw.npy --shot-table npy/shot_table.npz \
    --ped calib/ped.npy --gain calib/gain.npy --status-bad calib/status_bad.npy \
    --photon-kev 9.6 --out-dir outputs/<run>/selection/shot_conformity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FRAME_SHAPE = (2, 512, 1024)
NPIX = int(np.prod(FRAME_SHAPE))


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


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    if window <= 1 or window >= len(x):
        return np.full_like(x, np.median(x))
    return median_filter(x, size=window, mode="nearest")


def detrend_flux_then_time(x: np.ndarray, mono: np.ndarray, window: int):
    """Remove the flux-conditional expectation (rolling median in monitor order —
    absorbs monitor nonlinearity vs flux level), then the slow time drift.
    Returns (residual, flux_trend_in_time_order)."""
    order = np.argsort(mono, kind="stable")
    trend_f = np.empty_like(x)
    trend_f[order] = rolling_median(x[order], window)
    resid = x - trend_f
    resid = resid - rolling_median(resid, window)
    return resid, trend_f


def run(args) -> int:
    t = np.load(args.shot_table)
    ipm2, xray = t["ipm2"], t["xray"]
    offset = float(np.median(ipm2[xray == 0]))
    ipm2c = ipm2 - offset
    keep = (xray == 1) & (ipm2c >= args.low_ipm_threshold)
    idx = np.where(keep)[0]
    T = len(idx)
    mono = ipm2c[idx].astype(np.float64)
    f = mono / mono.mean()

    ped = np.load(args.ped)
    gain = np.load(args.gain)
    status_bad = np.load(args.status_bad).astype(bool).reshape(-1) if args.status_bad \
        else np.zeros(NPIX, bool)
    frames = np.load(args.frames, mmap_mode="r")

    # single calibration pass: dense lambda accumulator + sparse per-shot counts
    sum_counts = np.zeros(NPIX, np.float64)
    sparse_idx, sparse_k = [], []
    t0 = time.time()
    for n, i in enumerate(idx):
        kev = calibrate(np.asarray(frames[i]), ped, gain).reshape(-1)
        counts = np.maximum(np.rint(kev / args.photon_kev), 0.0)
        nz = np.flatnonzero(counts)
        sparse_idx.append(nz.astype(np.int32))
        sparse_k.append(counts[nz].astype(np.int32))
        sum_counts[nz] += counts[nz]
        if (n + 1) % 250 == 0:
            print(f"[shot_conformity] {n+1}/{T} shots ({time.time()-t0:.0f}s)", flush=True)

    lam = sum_counts / T
    healthy = (~status_bad) & (lam > 0)
    n_h = int(healthy.sum())
    lam_h_sum = float(lam[healthy].sum())

    # per-pixel spike thresholds on a log-lambda grid (upper edge -> conservative)
    from scipy.stats import poisson

    lam_ok = lam[healthy]
    grid = np.logspace(np.log10(max(lam_ok.min(), 1e-6)), np.log10(lam_ok.max()), 25)
    gbin = np.minimum(np.searchsorted(grid, lam), len(grid) - 1)  # per-pixel grid slot

    S = np.zeros(T)
    dev = np.zeros(T)
    n_spike = np.zeros(T, np.int64)
    top_spikes = []
    for n in range(T):
        pi, k = sparse_idx[n], sparse_k[n].astype(np.float64)
        hsel = healthy[pi]
        pi, k = pi[hsel], k[hsel]
        li = lam[pi]
        S[n] = k.sum()
        f_self = S[n] / lam_h_sum  # shot's own brightness, not the monitor's
        if f_self <= 0:
            continue
        mu = li * f_self
        dev_pos = 2.0 * (k * np.log(k / mu) - (k - mu)).sum()
        dev_zero = 2.0 * f_self * (lam_h_sum - li.sum())  # k=0 pixels contribute 2*mu
        dev[n] = (dev_pos + dev_zero) / n_h
        thr = poisson.isf(args.tail_p, grid * f_self)[gbin[pi]]
        spikes = k > thr
        n_spike[n] = int(spikes.sum())
        for j in np.where(spikes)[0][:5]:
            top_spikes.append({"shot": int(idx[n]), "pixel_flat": int(pi[j]),
                               "k": int(k[j]), "lambda": float(li[j]),
                               "tail_bound": float(thr[j])})

    M = f * lam_h_sum
    r = S / M - 1.0

    def mad(x):
        return float(np.median(np.abs(x - np.median(x))) * 1.4826)

    # detrend along flux (monitor nonlinearity) then time (drift), standardize r
    # by a combined multiplicative + Poisson error model, fence on the population
    rd, r_flux_trend = detrend_flux_then_time(r, mono, args.detrend_window)
    dd, d_flux_trend = detrend_flux_then_time(dev, mono, args.detrend_window)
    bright_half = M > np.median(M)  # Poisson part negligible here
    sigma_mult = mad(rd[bright_half])
    sigma_t = np.sqrt(sigma_mult**2 + 1.0 / M)
    u = rd / sigma_t

    u_mad, d_mad = mad(u), mad(dd)
    u_fence = args.fence_k * u_mad
    d_fence = float(np.median(dd) + args.fence_k * d_mad)

    flag_u = np.abs(u) > u_fence
    flag_d = dd > d_fence
    flag_s = n_spike >= args.spike_min
    flags = flag_u | flag_d | flag_s

    classes = np.full(T, "", object)
    classes[flag_d] = "structural"
    classes[flag_u & (u > 0)] = "excess_vs_monitor"
    classes[flag_u & (u < 0)] = "deficit_vs_monitor"
    classes[flag_s] = "zinger_spike"  # highest precedence

    # monitor-nonlinearity evidence: flux-binned median of the S/M ratio
    qedges = np.quantile(mono, np.linspace(0, 1, 21))
    ratio_curve = []
    for a, b in zip(qedges[:-1], qedges[1:]):
        s = (mono >= a) & (mono <= b)
        if s.sum() >= 10:
            ratio_curve.append({"ipm2c_range": [float(a), float(b)],
                                "n": int(s.sum()),
                                "ratio_median": float(np.median(1.0 + r[s]))})

    # the point of the method: bright shots are kept unless they fail conformity
    p99 = np.quantile(mono, 0.99)
    bright = mono > p99
    report = {
        "n_events": int(len(xray)), "n_xray_on": int((xray == 1).sum()),
        "low_ipm_threshold": args.low_ipm_threshold, "n_shots_screened": T,
        "monitor_cv": float(np.std(mono) / np.mean(mono)), "ipm2_offset": offset,
        "n_healthy_pixels": n_h, "lambda_sum": lam_h_sum,
        "mean_photons_per_shot": float(S.mean()),
        "sigma_mult": sigma_mult,
        "poisson_rel_floor": float(1.0 / np.sqrt(M.mean())),
        "ratio_vs_flux_curve": ratio_curve,
        "fences": {"fence_k": args.fence_k, "u_mad": u_mad, "u_fence": u_fence,
                   "d_mad": d_mad, "d_fence": d_fence,
                   "detrend_window": args.detrend_window,
                   "tail_p": args.tail_p, "spike_min": args.spike_min},
        "n_flagged": int(flags.sum()),
        "flag_fraction": float(flags.mean()),
        "class_counts": {c: int((classes == c).sum())
                         for c in np.unique(classes[flags]).tolist()},
        "bright_tail_demo": {
            "p99_ipm2c": float(p99), "n_above_p99": int(bright.sum()),
            "n_above_p99_flagged": int((bright & flags).sum()),
            "note": "conforming bright shots are KEPT - contrast with a fixed p99 cut",
        },
        "spike_examples": top_spikes[:20],
        "flagged_shots": [
            {"shot": int(idx[n]), "ipm2c": float(mono[n]), "f": float(f[n]),
             "S": float(S[n]), "M": float(M[n]), "ratio": float(1.0 + r[n]),
             "u": float(u[n]), "dev_per_px": float(dev[n]),
             "dev_detrended": float(dd[n]), "n_spike": int(n_spike[n]),
             "class": str(classes[n])}
            for n in np.where(flags)[0]
        ],
        "rejected_shot_indices": [int(s) for s in idx[flags]],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shot_conformity_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "shot_conformity.npz", shot=idx, ipm2c=mono, f=f,
                        S=S, M=M, ratio=1.0 + r, u=u, dev_per_px=dev,
                        dev_detrended=dd, n_spike=n_spike, flags=flags,
                        classes=classes.astype(str))
    _plots(out, mono, S, M, r, r_flux_trend, u, dd, n_spike, flags,
           u_fence, d_fence, p99)
    print(json.dumps({k: report[k] for k in
                      ["n_shots_screened", "monitor_cv", "sigma_mult",
                       "poisson_rel_floor", "n_flagged", "class_counts",
                       "bright_tail_demo"]}, indent=2))
    return 0


def _plots(out, mono, S, M, r, r_flux_trend, u, dd, n_spike, flags,
           u_fence, d_fence, p99):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    order = np.argsort(mono)
    ax.semilogx(mono, 1.0 + r, ".", ms=2, alpha=0.35, label="shots")
    ax.semilogx(mono[order], 1.0 + r_flux_trend[order], "-", color="orange",
                lw=1.5, label="flux-conditional median")
    if flags.any():
        ax.semilogx(mono[flags], 1.0 + r[flags], "rx", ms=7, label="flagged")
    ax.axvline(p99, color="gray", ls="--", lw=1, label="ipm2 p99 (old fixed cut)")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("ipm2c")
    ax.set_ylabel(r"ratio $S_t / M_t$")
    ax.set_title("detector/monitor ratio vs flux (monitor nonlinearity = curve)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(u, ".", ms=2, alpha=0.4)
    if flags.any():
        ax.plot(np.where(flags)[0], u[flags], "rx", ms=7)
    for s in (+1, -1):
        ax.axhline(s * u_fence, color="g", lw=1)
    ax.set_xlabel("shot (time order)")
    ax.set_ylabel(r"$u_t$  [$\sigma_t$ units]")
    ax.set_title("standardized conditional residual (fences green)")

    ax = axes[1, 0]
    ax.plot(dd, ".", ms=2, alpha=0.4)
    if flags.any():
        ax.plot(np.where(flags)[0], dd[flags], "rx", ms=7)
    ax.axhline(d_fence, color="g", lw=1)
    ax.set_xlabel("shot (time order)")
    ax.set_ylabel("deviance/pixel, conditionally detrended")
    ax.set_title("per-shot Poisson deviance (fence green)")

    ax = axes[1, 1]
    ax.semilogy(mono, np.maximum(n_spike, 0.5), ".", ms=3, alpha=0.5)
    if flags.any():
        ax.semilogy(mono[flags], np.maximum(n_spike[flags], 0.5), "rx", ms=7)
    ax.axvline(p99, color="gray", ls="--", lw=1, label="ipm2 p99 (old fixed cut)")
    ax.set_xlabel("ipm2c")
    ax.set_ylabel("n_spike (0 shown at 0.5)")
    ax.set_title("tail spikes vs shot intensity — bright conforming shots kept")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "shot_conformity.png", dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--shot-table", required=True)
    p.add_argument("--ped", required=True)
    p.add_argument("--gain", required=True)
    p.add_argument("--status-bad")
    p.add_argument("--photon-kev", type=float, required=True,
                   help="from the method file; no default on purpose")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--low-ipm-threshold", type=float, default=200.0,
                   help="selection 01a's cut; this method runs after it")
    p.add_argument("--fence-k", type=float, default=6.0)
    p.add_argument("--tail-p", type=float, default=1e-12)
    p.add_argument("--spike-min", type=int, default=1)
    p.add_argument("--detrend-window", type=int, default=101)
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
