#!/usr/bin/env python3
"""
methods.py -- automated masking pipeline for the xppl1016922 Jungfrau1M detector.

Runs in assembled space (1064, 1030) on the frozen dataset in ./data.

Lay down an intensity-free 100%-precision floor (geometry + psana calib), then
union three TV-regularized intensity detectors on top:

    combo = geometry | calib | variance | window_median | blackhat

Each detector is three separable stages:
  1. STATISTICS   -- *_stat(): raw arrays -> continuous per-pixel z-score field.
  2. TV           -- tv_denoise(): smooth that field before thresholding.
  3. COMBINATION  -- threshold_stat() -> boolean pick; combine_masks() unions.

Run:  python methods.py     (from src/automask/)
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import black_tophat, disk
from skimage.restoration import denoise_tv_chambolle
import matplotlib
# Preserve headless batch behavior, but allow review.py to select an interactive
# backend through MPLBACKEND (e.g. TkAgg or QtAgg).
if not os.environ.get("MPLBACKEND") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_image, load_mask, score

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
MASK_DIR = os.path.join(HERE, "outputs", "masks")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(MASK_DIR, exist_ok=True)

RUN = 475
PAD = 2                    # half-width for padding sparse masks (5x5 dilation)
CLEAN = 9                  # close_open square size (legacy baseline for studies/)
KVAR, WVAR = 3.5, 4.0      # variance: MAD threshold + TV weight
KWIN, WWIN = 5.0, 1.0      # window median: MAD threshold + TV weight
KBH, WBH = 6.0, 1.0        # black-hat: MAD threshold + TV weight
KSUM = 3.5                 # weighted-sum fusion: single threshold on the summed z-field


# ==========================================================================
#  floor masks -- geometry + calibration (intensity-free, 100%-precision)
# ==========================================================================
def geometry_mask(real: np.ndarray, pad: int = 2, frac: float = 0.4) -> np.ndarray:
    """Mask assembled border lines: rows/cols where more than `frac` of pixels
    are UNMAPPED (ASIC boundaries + inter-module gap), widened by `pad` px each
    side. `real` is the boolean map of pixels carrying a real value (sum != 0)."""
    H, W = real.shape
    mask = np.zeros((H, W), bool)
    mask[(~real).mean(1) > frac, :] = True
    mask[:, (~real).mean(0) > frac] = True
    return pad_mask(mask, pad)


_CALIB_MASK_BY_RUN = {389: "statusMask_run0389", 475: "statusMask_run0475"}


def fetch_calib_mask(run: int) -> np.ndarray:
    """psana calib-store bad-pixel mask (True == masked), the second
    100%-precision-floor ingredient alongside geometry_mask. This is
    `pixel_status`, frozen per run into data/masks/statusMask_run<NNNN>_asm.npy.

    TODO: hardcoded to the two runs frozen in ./data. Generalize by selecting the
    right pixel_status <START>-end.data file for an arbitrary run from calib/."""
    try:
        name = _CALIB_MASK_BY_RUN[run]
    except KeyError:
        raise ValueError(
            f"no frozen calibration mask for run {run}; only "
            f"{sorted(_CALIB_MASK_BY_RUN)} are available") from None
    return load_mask(name)


# ==========================================================================
#  1. statistics -- continuous per-pixel evidence fields, no thresholding
# ==========================================================================
def robust_z(field, domain, transform=None):
    """Signed robust-MAD z-score of `field` over boolean `domain`, 0 outside it.
    `transform` (if given) is applied AFTER indexing into `domain`."""
    v = field[domain]
    if transform is not None:
        v = transform(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    z = np.zeros(field.shape, dtype=np.float64)
    z[domain] = (v - med) / (mad + 1e-12)
    return z


def variance_stat(ustd):
    """Signed robust-MAD z-score of log10(per-pixel std `ustd`).

    z << 0 = low-variance (dead/shadowed/beam-stop), z >> 0 = high-variance. On a
    lit run photon shot noise lifts live pixels, so defects show as LOW-variance
    islands -- method_variance thresholds mode="low"."""
    return robust_z(ustd, ustd > 0, transform=np.log10)


def method_variance(ustd, real, k=KVAR, weight=WVAR, mode="low"):
    """TV-denoise variance_stat() then threshold (mode="low")."""
    u = tv_denoise(variance_stat(ustd), weight)
    return threshold_stat(u, k, mode) & real


def window_median_stat(lit, real, win=21):
    """Local tile-median-subtraction residual as a signed robust-MAD z-score.
    Tiles the frame into win x win blocks, subtracts each block's median (over
    `real` pixels) to flatten slow illumination, then z-scores the residual.
    `lit` MUST be a genuine lit-beam frame (umean), not the dark `mean`."""
    H, W = lit.shape
    R = np.zeros((H, W), dtype=np.float64)
    for i in range(0, H, win):
        i1 = min(i + win, H)
        for j in range(0, W, win):
            j1 = min(j + win, W)
            tile, tr = lit[i:i1, j:j1], real[i:i1, j:j1]
            v = tile[tr]
            R[i:i1, j:j1] = tile - (np.median(v) if v.size else 0.0)
    return robust_z(R, real)


def method_window_median(lit, real, win=21, k=KWIN, pad=PAD, mode="low", *, weight=WWIN):
    """TV-denoise window_median_stat() then threshold (mode="low") and pad.
    `lit` must be a lit-beam frame (umean), not the dark `mean`. Hits are sparse,
    so padding is load-bearing -- TV smooths the field, pad grows the picks."""
    u = tv_denoise(window_median_stat(lit, real, win=win), weight)
    picked = threshold_stat(u, k, mode) & real
    return pad_mask(picked, pad) & real


def blackhat_stat(lit, real, radius=5):
    """Black-hat response as a robust-MAD z-score: closing(lit) - lit over `real`.
    Responds to pixels DARKER than their local surroundings (dead spots, dark
    specks, beam-stop edge). One-sided (R >= 0), so threshold mode="high"."""
    filled = np.where(real, lit, np.median(lit[real]))
    R = black_tophat(filled, footprint=disk(radius))
    return robust_z(R, real)


def method_blackhat(lit, real, radius=5, k=KBH, weight=WBH, pad=PAD):
    """TV-denoise blackhat_stat() then threshold (mode="high") and pad.
    `lit` must be a lit-beam frame (umean). Hits are sparse specks, so padding
    supplies the area TV does not."""
    u = tv_denoise(blackhat_stat(lit, real, radius=radius), weight)
    M = threshold_stat(u, k, mode="high") & real
    return pad_mask(M, pad) & real


def wavelet_stat(lit, real, sigmas=(1,), win=15):
    """Directional Ricker (Mexican-hat) wavelet line/point SNR, unsigned (>= 0).
    Row/column order-2 gaussian passes respond to lines and isolated pixels;
    dividing by the local scale turns the response into an SNR. Sees BRIGHT
    defects, so it is a DARK-run tool -- not in the production combo."""
    filled = np.where(real, lit, np.median(lit[real]))
    Z = np.zeros_like(filled)
    for s in sigmas:
        for axis in (0, 1):
            R = np.abs(s * s * ndi.gaussian_filter1d(filled, s, axis=axis, order=2))
            Z = np.maximum(Z, R / (ndi.median_filter(R, size=win) + 1e-9))
    return Z


def method_wavelet(lit, real, sigmas=(1,), k=12.0, win=15):
    """Threshold wavelet_stat() (mode="high"), unpadded. Dark-run tool, not in
    combo; re-tune `k` before use on a dark run."""
    return threshold_stat(wavelet_stat(lit, real, sigmas=sigmas, win=win), k, "high") & real


# ==========================================================================
#  2. regularization (TV)
# ==========================================================================
def tv_denoise(z, weight):
    """Isotropic Total-Variation denoising (Chambolle):
        u = argmin_u  ||u - z||^2 + weight * TV(u)
    weight <= 0 is a no-op passthrough."""
    return denoise_tv_chambolle(z, weight=weight) if weight > 0 else z


# ==========================================================================
#  3. combination -- threshold each statistic, then union into a mask
# ==========================================================================
def threshold_stat(z, k, mode="low"):
    """Threshold a continuous field: "low" -> z < -k, "high" -> z > k, "both" ->
    either. Callers gate by `real`/domain themselves."""
    hi, lo = (z > k), (z < -k)
    return {"low": lo, "high": hi, "both": hi | lo}[mode]


def pad_mask(mask, pad=PAD):
    """Grow a boolean mask by `pad` px each side with a (2*pad+1) square dilation."""
    if not pad:
        return mask
    return ndi.binary_dilation(mask, structure=np.ones((2 * pad + 1, 2 * pad + 1)))


def close_open(mask, size=CLEAN):
    """Binary close then open with a (size, size) square. Superseded by TV; kept
    for the studies/ that compare against it. Dense masks only."""
    se = np.ones((size, size), bool)
    return ndi.binary_opening(ndi.binary_closing(mask, structure=se), structure=se)


def combine_masks(floor, picks):
    """Union the floor with every per-method pick. The ONLY place union logic
    lives, so a smarter future combiner replaces just this function. `picks` is a
    dict[str, np.ndarray] for readability; the keys are not otherwise used."""
    combined = floor
    for m in picks.values():
        combined = combined | m
    return combined


def defectiveness_fields(ustd, umean, real):
    """Sign-aligned, TV-denoised DEFECTIVENESS fields (large > 0 == wants masking),
    the common substrate the weighted-sum combiner fuses. Variance and window-median
    flag LOW z, black-hat flags HIGH z; we fold the first two so all three point the
    same way and share one robust-z scale. Fields are 0 outside `real`.

    Keys mirror the per-method picks in main() so combine_stats() and combine_masks()
    consume interchangeable dicts. `umean` must be a lit-beam frame, not the dark mean."""
    d_var = tv_denoise(-variance_stat(ustd), WVAR)
    d_wm  = tv_denoise(-window_median_stat(umean, real, win=21), WWIN)
    d_bh  = tv_denoise(blackhat_stat(umean, real, radius=5), WBH)
    for d in (d_var, d_wm, d_bh):
        d[~real] = 0.0
    return {"variance": d_var, "window_median": d_wm, "blackhat": d_bh}


def combine_stats(floor, fields, real, weights=None, k=KSUM, pad=PAD):
    """Weighted-sum fusion combiner -- the joint alternative to combine_masks().

    Instead of thresholding each statistic on its own and unioning the booleans,
    sum the aligned defectiveness `fields` into ONE score S = sum_i w_i d_i and
    threshold it a single time, then union the pick onto the intensity-free floor.
    `weights` default to equal (the fields already share a robust-z scale). Padding
    grows the sparse pick, matching the per-method combiners.
    """
    names = list(fields)
    w = np.ones(len(names)) if weights is None else np.asarray(weights, float)
    S = np.zeros(floor.shape, dtype=np.float64)
    for wi, n in zip(w, names):
        S += wi * fields[n]
    picked = pad_mask((S > k) & real, pad) & real
    return floor | picked


def mask_image(
    image: np.ndarray,
    *,
    win: int = 21,
    window_k: float = KWIN,
    window_weight: float = WWIN,
    blackhat_radius: int = 5,
    blackhat_k: float = KBH,
    blackhat_weight: float = WBH,
    pad: int = PAD,
) -> np.ndarray:
    """Single-image adapter used by :mod:`review`.

    This is the honest subset of the run-level pipeline that can operate on one
    arbitrary 2-D diffraction image: invalid pixels and geometry lines form the
    floor, then window-median and black-hat picks are unioned with it.  The
    variance detector is intentionally absent because per-pixel variance cannot
    be estimated from one frame.  The run-specific psana calibration mask is
    also absent because the public corpus mixes facilities and detectors.

    Parameters are exposed so an experiment-specific adapter can call this
    function with tuned values.  The reviewer itself calls it with the defaults
    when invoked as ``--masker methods:mask_image``.

    Mask convention is boolean ``True == excluded/masked``.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"mask_image expects a 2-D array, got {image.shape}")
    work = image.astype(np.float64, copy=False)
    finite = np.isfinite(work)
    if not finite.any():
        return np.ones(image.shape, dtype=bool)

    # geometry_mask needs a map of detector pixels carrying data. Exact zeros
    # are useful for finding full panel gaps, but are not masked individually:
    # sparse photon-counting frames can contain many legitimate zero pixels.
    carrying_data = finite & (work != 0)
    floor = ~finite | geometry_mask(carrying_data, pad=pad)
    window_median = method_window_median(
        work, finite, win=win, k=window_k, pad=pad, weight=window_weight
    )
    blackhat = method_blackhat(
        work, finite, radius=blackhat_radius, k=blackhat_k,
        weight=blackhat_weight, pad=pad
    )
    return combine_masks(
        floor,
        {"window_median": window_median, "blackhat": blackhat},
    ).astype(bool, copy=False)


# ==========================================================================
#  data
# ==========================================================================
def load_all(run):
    """Returns (sumimg, mean, ustd, human). NOTE: `mean` is a beam-OFF
    dark/pedestal frame, not a lit-beam average -- use load_umean() for methods
    that need real scattering contrast."""
    sumimg = load_image(f"sum_calib_run{run:04d}")
    f = os.path.join(HERE, "data", "features")
    mean = np.load(os.path.join(f, f"mean_run{run:04d}_asm.npy")).astype(np.float64)
    ustd = np.load(os.path.join(f, f"ustd_run{run:04d}_asm.npy")).astype(np.float64)
    # Prefer a run-specific target (lab recipe re-run on this run's frames); fall
    # back to the shared 475-built human_Mask.
    per_run = os.path.join(HERE, "data", "masks", f"human_Mask_run{run:04d}_asm.npy")
    human = (np.load(per_run).astype(bool) if os.path.exists(per_run)
             else load_mask("human_Mask"))
    return sumimg, mean, ustd, human


def load_umean(run):
    """Genuine lit-beam per-pixel mean (producers/normalized_median.py's `umean`)."""
    f = os.path.join(HERE, "data", "features")
    return np.load(os.path.join(f, f"umean_run{run:04d}_asm.npy")).astype(np.float64)


# ==========================================================================
#  figures
# ==========================================================================
def _agree(pred, truth):
    a = np.ones((*truth.shape, 3))
    a[pred & truth] = (0.0, 0.7, 0.0)       # TP green
    a[pred & ~truth] = (0.9, 0.0, 0.0)      # FP red
    a[~pred & truth] = (0.0, 0.3, 1.0)      # FN blue
    return a


def plot_pipeline(floor, var_res, wm_res, bhat_res, combo, human, run):
    """2x3 panel: geometry+calib floor, the three TV detectors' residual picks,
    the combined mask, and the combo-vs-human error map."""
    bw = mcolors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(2, 3, figsize=(17, 11))

    def show(a, m, title):
        a.imshow(m, cmap=bw); a.set_title(title, fontsize=11); a.axis("off")

    target = human & ~floor
    sf, sc = score(floor, human), score(combo, human)
    sv, sw, sb = (score(var_res, target), score(wm_res, target),
                  score(bhat_res, target))

    show(ax[0, 0], floor, f"geometry + calib floor ({100*floor.mean():.1f}%)\n"
                          f"IoU={sf['iou']:.3f} prec={sf['precision']:.2f}")
    show(ax[0, 1], var_res, f"variance (TV), residual\n"
                            f"T-prec={sv['precision']:.2f} T-rec={sv['recall']:.3f}")
    show(ax[0, 2], wm_res, f"window median (TV), residual\n"
                           f"T-prec={sw['precision']:.2f} T-rec={sw['recall']:.3f}")
    show(ax[1, 0], bhat_res, f"black-hat (TV), residual\n"
                             f"T-prec={sb['precision']:.2f} T-rec={sb['recall']:.3f}")
    show(ax[1, 1], combo, f"combined mask ({100*combo.mean():.1f}%)\n"
                          f"IoU={sc['iou']:.3f} prec={sc['precision']:.2f}")

    ax[1, 2].imshow(_agree(combo, human))
    ax[1, 2].set_title(f"error: combo vs human\n"
                       f"green=TP red=FP blue=FN (rec={sc['recall']:.2f})", fontsize=11)
    ax[1, 2].axis("off")

    fig.suptitle(f"xppl1016922 run {run} — geometry+calib floor, then TV "
                 f"variance + window-median + black-hat", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, f"methods_run{run:04d}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


# ==========================================================================
#  main
# ==========================================================================
def main():
    sumimg, _mean, ustd, human = load_all(RUN)
    umean = load_umean(RUN)
    real = sumimg != 0

    calib = fetch_calib_mask(RUN)
    floor = geometry_mask(real) | calib
    target = human & ~floor
    sf = score(floor, human)
    print(f"=== geometry + calib floor (run {RUN}) ===")
    print(f"  floor vs human : {100*floor.mean():.2f}% masked  IoU {sf['iou']:.3f}  "
          f"prec {sf['precision']:.3f}  rec {sf['recall']:.3f}")
    print(f"  residual target = human & ~floor : {int(target.sum())} px to find\n")

    variance = method_variance(ustd, real)
    window_median = method_window_median(umean, real)
    blackhat = method_blackhat(umean, real)
    combo = combine_masks(floor, {"variance": variance,
                                  "window_median": window_median,
                                  "blackhat": blackhat})
    # Joint weighted-sum fusion of the same three detectors' continuous fields,
    # as an alternative to the union combo above (see combine_stats docstring).
    combo_sum = combine_stats(floor, defectiveness_fields(ustd, umean, real), real)

    print(f"{'detector':16s} | {'added%':>6s} {'T-prec':>6s} {'T-rec':>6s} "
          f"| {'IoU':>7s} {'prec':>6s} {'rec':>6s}")
    print("-" * 70)
    for name, M in {"variance": variance, "window_median": window_median,
                    "blackhat": blackhat, "combo": combo,
                    "combo_sum": combo_sum}.items():
        Mo = M & ~floor
        st, sc = score(Mo, target), score(floor | M, human)
        print(f"{name:16s} | {100*Mo.mean():5.2f}% {st['precision']:6.3f} "
              f"{st['recall']:6.3f} | {sc['iou']:7.3f} {sc['precision']:6.3f} "
              f"{sc['recall']:6.3f}")
    print(f"\n(reference: floor alone -> IoU {sf['iou']:.3f})")

    np.save(os.path.join(MASK_DIR, f"geometry_mask_run{RUN:04d}.npy"), floor)
    np.save(os.path.join(MASK_DIR, f"combo_run{RUN:04d}.npy"), combo)
    np.save(os.path.join(MASK_DIR, f"combo_sum_run{RUN:04d}.npy"), combo_sum)

    plot_pipeline(floor, variance & ~floor, window_median & ~floor,
                  blackhat & ~floor, combo, human, RUN)


if __name__ == "__main__":
    main()
