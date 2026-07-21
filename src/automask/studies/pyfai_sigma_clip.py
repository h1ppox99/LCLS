#!/usr/bin/env python3
"""
pyfai_sigma_clip.py -- mask obtained from pyFAI's azimuthal sigma-clipping
alone, swept over the clip threshold k, plus a hand-rolled reimplementation
of the same algorithm compared against it.

The geometry mask (ASIC boundaries + inter-module gap, methods.geometry_mask)
is pre-applied: it is excluded from both methods' ring statistics and OR'd
back into every output mask (it is a 100%-precision floor, see methods.py).

pyFAI version: build an AzimuthalIntegrator for the Jungfrau1M using the
fixed geometry from CLAUDE.md (dist=190mm, wavelength=1.2915A, 75um pixels)
and the run's beam center (geometry.get_center -- see that module's
docstring for why this is NOT simply CLAUDE.md's "(col 1005, row 45)"
plugged in directly: the frozen `_asm` arrays store axis0/axis1 transposed
relative to that phrasing), and run `ai.sigma_clip_ng` on the lit-beam run-sum
image (`sumimg` from load_all -- NOT `load_all`'s `mean`, which is the beam-OFF
dark/pedestal frame (`_dropped`/N, see ARCHIVE.md) and carries no diffraction
signal at all; sigma-clipping it can never find a beam-dependent shadow) --
pyFAI's own iterative sigma-clipped azimuthal average (thres=k, max_iter=5).
That gives a robust <I(q)> and sigma(q) per radial bin (via its LUT/CSR
engine); interpolating both back onto every pixel's q gives a per-pixel
z-score, and |z| > k is flagged. z-scores are scale-invariant, so using the
run-sum directly (rather than rescaling to a per-shot mean) does not change
any mask.

Dummy (by-hand) version: same idea, no pyFAI engine. q(row,col) is computed
directly from the pinhole-camera geometry, pixels are digitized into NPT
q-bins, and the per-bin mean/std are recomputed with plain np.bincount,
iteratively dropping pixels with |I - bin-mean| > k*bin-std (max_iter passes)
-- the textbook sigma-clip loop. See hand_sigma_clip_mask().

Same k drives both methods' internal clipping and their final per-pixel cut.

Outputs:
    outputs/figures/pyfai_<RUN>.png
Run:  python pyfai_sigma_clip.py     (from src/automask/)
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

AUTOMASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../automask
sys.path.insert(0, AUTOMASK)
from dataset import score
from methods import geometry_mask, load_all
from geometry import get_center

FIG_DIR = os.path.join(AUTOMASK, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUN = 475
K_VALUES = (1.25, 1.5, 1.75 , 2.0)
NPT = 300

# fixed Jungfrau1M geometry, see CLAUDE.md "Detectors & geometry"
PIXEL_M = 75e-6
DIST_M = 0.190
WAVELENGTH_M = 1.2915e-10


def pyfai_sigma_clip_mask(mean, cand_mask, ai, k, npt=NPT):
    """pyFAI sigma-clipping: robust <I(q)>, sigma(q) from ai.sigma_clip_ng,
    interpolated back onto the pixel grid, |z| > k flagged.
    `cand_mask` (True == excluded) is both the pyFAI input mask and the set
    of pixels never eligible to be flagged (geometry + detector gaps)."""
    result = ai.sigma_clip_ng(mean, npt=npt, unit="q_nm^-1", mask=cand_mask,
                               error_model="azimuthal", thres=k, max_iter=5,
                               correctSolidAngle=False)
    qmap = ai.array_from_unit(mean.shape, unit="q_nm^-1", scale=True)
    mu = np.interp(qmap.ravel(), result.radial, result.intensity).reshape(mean.shape)
    sigma = np.interp(qmap.ravel(), result.radial, result.std).reshape(mean.shape)
    z = np.zeros_like(mean)
    ok = ~cand_mask & (sigma > 0)
    z[ok] = (mean[ok] - mu[ok]) / sigma[ok]
    return np.abs(z) > k


def q_map_by_hand(shape, center0, center1) -> np.ndarray:
    """q [nm^-1] at every assembled pixel from plain pinhole-camera geometry
    -- no pyFAI. q = (4*pi/lambda) * sin(2theta/2), 2theta = atan(r/dist).
    `center0`/`center1` are in the array's own (axis0, axis1) index order --
    see geometry.get_center -- NOT a (row, col) pair."""
    i0, i1 = np.indices(shape)
    r = np.hypot((i0 - center0) * PIXEL_M, (i1 - center1) * PIXEL_M)
    two_theta = np.arctan2(r, DIST_M)
    q_per_m = 4 * np.pi / WAVELENGTH_M * np.sin(two_theta / 2)
    return q_per_m * 1e-9


def hand_sigma_clip_mask(mean, cand_mask, qmap, k, npt=NPT, max_iter=5):
    """Dummy hand-rolled sigma-clipping: bin candidate pixels into `npt`
    q-bins, then iteratively recompute each bin's mean/std with
    np.bincount and drop pixels more than k*std from their bin's mean --
    the same loop pyFAI runs internally, just spelled out with plain numpy
    instead of its LUT/CSR integration engine. Final mean/std (fit on the
    LAST iteration's surviving pixels) are mapped back to every candidate
    pixel by its bin index, giving a per-pixel z-score; |z| > k is flagged."""
    cand = ~cand_mask
    edges = np.linspace(qmap[cand].min(), qmap[cand].max() + 1e-9, npt + 1)
    bidx = np.clip(np.digitize(qmap, edges) - 1, 0, npt - 1)

    included = cand.copy()
    for _ in range(max_iter):
        cnt = np.bincount(bidx[included], minlength=npt).astype(np.float64)
        sm = np.bincount(bidx[included], weights=mean[included], minlength=npt)
        sq = np.bincount(bidx[included], weights=mean[included] ** 2, minlength=npt)
        mu_bin = np.divide(sm, cnt, out=np.zeros(npt), where=cnt > 0)
        var_bin = np.divide(sq, cnt, out=np.zeros(npt), where=cnt > 0) - mu_bin ** 2
        std_bin = np.sqrt(np.clip(var_bin, 0, None))

        mu_px, std_px = mu_bin[bidx], std_bin[bidx]
        new_included = cand & (np.abs(mean - mu_px) <= k * std_px)
        if np.array_equal(new_included, included):
            break
        included = new_included

    z = np.zeros_like(mean)
    ok = cand & (std_px > 0)
    z[ok] = (mean[ok] - mu_px[ok]) / std_px[ok]
    return np.abs(z) > k


def main():
    sumimg, _mean, ustd, human = load_all(RUN)
    real = sumimg != 0

    geom = geometry_mask(real)                          # 100%-precision floor, pre-applied

    center0, center1 = get_center(RUN)
    ai = AzimuthalIntegrator(dist=DIST_M,
                              poni1=center0 * PIXEL_M, poni2=center1 * PIXEL_M,
                              pixel1=PIXEL_M, pixel2=PIXEL_M,
                              wavelength=WAVELENGTH_M)

    cand_mask = geom | ~real                            # excluded from both methods' ring stats
    qmap = q_map_by_hand(sumimg.shape, center0, center1)

    bw = mcolors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(2, len(K_VALUES), figsize=(5 * len(K_VALUES), 11))

    hdr = (f"{'k':>5s} | {'pyFAI masked%':>13s} {'IoU':>6s} {'prec':>6s} {'rec':>6s} "
           f"| {'hand masked%':>12s} {'IoU':>6s} {'prec':>6s} {'rec':>6s} | {'agree IoU':>9s}")
    print(hdr); print("-" * len(hdr))
    for col, k in enumerate(K_VALUES):
        pyfai_out = pyfai_sigma_clip_mask(sumimg, cand_mask, ai, k)
        hand_out = hand_sigma_clip_mask(sumimg, cand_mask, qmap, k)
        pyfai_combo = geom | pyfai_out
        hand_combo = geom | hand_out
        sp, sh = score(pyfai_combo, human), score(hand_combo, human)
        agree = score(hand_combo, pyfai_combo)["iou"]     # how alike are the two masks
        print(f"{k:5.1f} | {100*pyfai_combo.mean():12.2f}% {sp['iou']:6.3f} "
              f"{sp['precision']:6.3f} {sp['recall']:6.3f} | "
              f"{100*hand_combo.mean():11.2f}% {sh['iou']:6.3f} "
              f"{sh['precision']:6.3f} {sh['recall']:6.3f} | {agree:9.3f}")

        ax[0, col].imshow(pyfai_combo, cmap=bw)
        ax[0, col].set_title(f"pyFAI  k={k:g}  ({100*pyfai_combo.mean():.1f}% masked)\n"
                              f"IoU={sp['iou']:.3f} prec={sp['precision']:.2f} rec={sp['recall']:.3f}",
                              fontsize=11)
        ax[0, col].axis("off")

        ax[1, col].imshow(hand_combo, cmap=bw)
        ax[1, col].set_title(f"hand   k={k:g}  ({100*hand_combo.mean():.1f}% masked)\n"
                              f"IoU={sh['iou']:.3f} prec={sh['precision']:.2f} rec={sh['recall']:.3f}  "
                              f"(agree w/ pyFAI={agree:.3f})",
                              fontsize=11)
        ax[1, col].axis("off")

    fig.suptitle(f"xppl1016922 run {RUN} — geometry mask + sigma-clip alone, vs k\n"
                 f"top: pyFAI (ai.sigma_clip_ng)   bottom: hand-rolled reimplementation",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=6)
    out = os.path.join(FIG_DIR, f"pyfai_{RUN}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
