# ==========================================================================
def defectiveness_fields(ustd, umean, real):
    """The three intensity detectors as TV-denoised DEFECTIVENESS fields:
    large positive == wants masking, 0 outside `real`. Sign-folded so variance
    and window-median (which flag LOW z) point the same way as black-hat.

    Returns a dict name -> (H, W) float field, all on a comparable robust-z scale.
    """
    d_var = tv_denoise(-variance_stat(ustd), WVAR)
    d_wm  = tv_denoise(-window_median_stat(umean, real, win=21), WWIN)
    d_bh  = tv_denoise(blackhat_stat(umean, real, radius=5), WBH)
    for d in (d_var, d_wm, d_bh):
        d[~real] = 0.0
    return {"variance": d_var, "window_median": d_wm, "blackhat": d_bh}


def stack(fields, domain):
    """(N, k) feature matrix of the k fields sampled over boolean `domain`."""
    return np.column_stack([f[domain] for f in fields.values()])


# ==========================================================================
#  fusers -> continuous score field over `domain`
# ==========================================================================
def fuse_weighted_sum(fields, domain, weights=None):
    """Linear vote S = sum_i w_i d_i. Equal weights by default (fields are all
    on a robust-z scale already). Higher == more defect-like."""
    names = list(fields)
    w = np.ones(len(names)) if weights is None else np.asarray(weights, float)
    S = np.zeros(next(iter(fields.values())).shape)
    for wi, n in zip(w, names):
        S += wi * fields[n]
    S[~domain] = -np.inf
    return S


def _trimmed_moments(X, keep_frac=0.8, iters=5):
    """Robust (hard-trimmed) mean + covariance a la a cheap MCD, numpy-only.
    Iteratively keep the `keep_frac` closest pixels in Mahalanobis distance and
    refit, so the estimate reflects the good-pixel bulk, not the defect tail."""
    mu = np.median(X, axis=0)
    cov = np.cov(X, rowvar=False)
    keep = np.ones(len(X), bool)
    for _ in range(iters):
        Xc = X - mu
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
        thr = np.quantile(d2, keep_frac)
        keep = d2 <= thr
        mu = X[keep].mean(axis=0)
        cov = np.cov(X[keep], rowvar=False)
    return mu, cov


def fuse_mahalanobis(fields, domain, keep_frac=0.8):
    """Joint-outlier score: robust Mahalanobis distance of each pixel's
    (d_var, d_wm, d_bh) vector from the good-pixel bulk. Uses the field
    correlations a weighted sum ignores. Returns sqrt(D^2) over `domain`."""
    X = stack(fields, domain)
    mu, cov = _trimmed_moments(X, keep_frac=keep_frac)
    inv = np.linalg.pinv(cov)
    Xc = X - mu
    d2 = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
    S = np.full(domain.shape, -np.inf)
    S[domain] = np.sqrt(d2)
    return S


# ==========================================================================
#  threshold sweep -> best residual-IoU operating point
# ==========================================================================
def best_threshold(S, domain, target, ks, pad=PAD):
    """Sweep threshold `ks` on score field S (mask = S>k, then pad & domain),
    return (k*, mask*, iou*) maximizing residual IoU vs `target`."""
    best = (ks[0], None, -1.0)
    for k in ks:
        M = pad_mask((S > k) & domain, pad) & domain
        iou = score(M, target)["iou"]
        if iou > best[2]:
            best = (k, M, iou)
    return best


# ==========================================================================
#  figures
# ==========================================================================
def _agree(pred, truth):
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)
    a[pred & ~truth] = (0.9, 0.0, 0.0)
    a[~pred & truth] = (0.0, 0.3, 1.0)
    return a


def plot_run(run, floor, target, union, fusers, curves, out):
    """Row of residual error maps (union baseline + each fuser at its best k),
    plus an IoU-vs-threshold panel."""
    n = 2 + len(fusers)
    fig, ax = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.4))

    su = score(union & ~floor, target)
    ax[0].imshow(_agree((union & ~floor), target)); ax[0].axis("off")
    ax[0].set_title(f"union combo (baseline)\nIoU={su['iou']:.3f} "
                    f"p={su['precision']:.2f} r={su['recall']:.2f}", fontsize=10)

    for i, (name, (k, M, iou)) in enumerate(fusers.items(), start=1):
        s = score(M, target)
        ax[i].imshow(_agree(M, target)); ax[i].axis("off")
        ax[i].set_title(f"{name} @k={k:.2f}\nIoU={s['iou']:.3f} "
                        f"p={s['precision']:.2f} r={s['recall']:.2f}", fontsize=10)

    bw = mcolors.ListedColormap(["white", "black"])
    ax[n - 1].imshow(target, cmap=bw); ax[n - 1].axis("off")
    ax[n - 1].set_title(f"residual target\n{int(target.sum())} px", fontsize=10)

    axc = ax[n]
    for name, (ks, ious) in curves.items():
        axc.plot(ks, ious, label=name, lw=1.6)
    axc.axhline(su["iou"], ls="--", c="k", lw=1.2, label="union combo")
    axc.set_xlabel("threshold k"); axc.set_ylabel("residual IoU")
    axc.set_title("IoU vs threshold"); axc.legend(fontsize=8); axc.grid(alpha=0.3)

    fig.suptitle(f"xppl1016922 run {run} — joint fusion vs union combo", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


# ==========================================================================
#  main