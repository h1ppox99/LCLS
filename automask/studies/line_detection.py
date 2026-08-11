#!/usr/bin/env python3
"""
studies/line_detection.py -- straight-line defect detection by Radon transform
and by probabilistic Hough segments, scored as a candidate production detector.

Motivation. The production pipeline (geometry+calib floor + TV variance, see
masking.production_pipeline) is *per-pixel*: every detector
decides pixel-by-pixel and knows nothing about shape. The defects it misses most
visibly are extended STRAIGHT structures -- wire/edge shadows, scratches, the
dark seams between ASIC groups, the beam-stop arm -- whose individual pixels sit
too close to the bulk to clear a robust-z bar, but which are unmistakable as a
line. Frangi (regularization/frangi.py) enhances such structure locally; the two
classical *global* line detectors are tried here instead:

    RADON  a full sinogram of the anomaly map. Every (angle, offset) bin counts
           the flagged pixels along one whole line of the detector, so a line
           whose pixels are individually unremarkable accumulates while scattered
           specks average out -- the highest-SNR way to find full-chip straight
           structure. Support-normalized (divided by the Radon transform of the
           domain indicator) so the bin value is the FRACTION of its chord that
           is flagged and lines crossing the module gap are not penalized, then
           robust-z'd across bins; peaks become thin bands back-projected into
           image space and trimmed to the span their own evidence covers.
    HOUGH  `probabilistic_hough_line` on the same anomaly map. Where Radon votes
           globally and needs the offset/angle grid to resolve the line, Hough
           returns SEGMENTS with endpoints directly and follows short, broken
           runs Radon's chord average would dilute -- but it needs a locally
           dense binary and cannot bridge a faint stretch.

Both consume the SAME input -- the black-hat darkness z-score of the run-sum
image (stats/blackhat.py, high == darker than the local surroundings) binarized
at `bin_k` and de-specked -- so the comparison is between the two line finders,
not between two evidence fields. Both are restricted to the residual domain
`real & ~production_mask`, so what they add is genuinely new.

Feeding Radon the CONTINUOUS field instead was tried first and fails: chords
through the bright corner/ring dominate every bin and the sinogram peaks trace
the sinusoid of a point source rather than the ridges of lines. Binarizing makes
one hot pixel worth 1 vote and a 500-px line worth 500.

Output, per evaluation run, is one 8-panel figure showing the mask you would get
if the detector were unioned into the production pipeline, plus a printed table
of IoU / precision / recall for production, +radon, +hough and +both.

Outcome: HOUGH won and has been promoted to `stats/hough_lines.py` (+0.031 IoU
on run 389 at 0.74 precision on the pixels it adds, and a clean abstention on
475). The study now imports the binarization and rasterization from there, so
the two cannot drift apart; what stays here is the comparison it was written to
settle -- RADON is kept unregistered because it needs a higher bar to be safe
(harmful below k~14) and buys less at lower precision than Hough.

Run:  python -m automask.studies.line_detection            # both eval runs
      python -m automask.studies.line_detection --sweep    # knob sensitivity
      python -m automask.studies.line_detection --selftest # geometry check only
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.draw import line as draw_line
from skimage.feature import peak_local_max
from skimage.transform import radon

from automask.dataset import score
from automask.evaluation import EVAL_RUNS, load_sample
from automask.masking import production_pipeline
from automask.stats.base import robust_z
from automask.stats.blackhat import blackhat_stat
from automask.stats.hough_lines import (anomaly_map as _anomaly_map,
                                        hough_segments, segments_mask)
from automask.viz import agree_rgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "outputs", "figures", "line_detection")
os.makedirs(OUTDIR, exist_ok=True)


# ==========================================================================
#  parameters
# ==========================================================================
@dataclass
class RadonParams:
    n_angles: int = 360      # sinogram angular sampling over [0, 180) deg
    k: float = 25.0          # robust-MAD z bar on the support-normalized sinogram
    min_support: float = 0.35  # keep bins whose chord covers >= this frac of the max
    min_hits: int = 40       # absolute evidence floor: flagged px on the chord
    min_distance: int = 6    # peak_local_max separation, in sinogram bins
    max_lines: int = 30      # safety cap on accepted lines
    width: float = 2.0       # back-projected band half-width, px  (~5px band)
    trim: bool = True        # clip each band to the span its own evidence covers
    trim_pad: float = 10.0   # slack added at both ends of that span, px


@dataclass
class HoughParams:
    threshold: int = 10      # Hough accumulator votes needed
    line_length: int = 100   # shortest segment kept, px
    line_gap: int = 5        # gap tolerated inside one segment, px
    width: int = 1           # dilation radius applied to rasterized segments, px


@dataclass
class Params:
    blackhat_radius: int = 5  # black-hat structuring element, px
    bin_k: float = 4.0        # robust-MAD z bar binarizing the darkness field
    min_size: int = 4         # drop connected specks below this many px
    radon: RadonParams = None
    hough: HoughParams = None

    def __post_init__(self):
        self.radon = self.radon or RadonParams()
        self.hough = self.hough or HoughParams()


# ==========================================================================
#  shared input: darkness field -> binary anomaly map
# ==========================================================================
def darkness_field(sample, radius: int = 5) -> np.ndarray:
    """Black-hat darkness z-score of the run-sum image: large == darker than the
    local surroundings. Line defects on this detector are dark (shadows, dead
    rows, scratches), which is why this is the field the line finders read."""
    return blackhat_stat(sample.sumimg, sample.real, radius=radius)


def anomaly_map(field: np.ndarray, domain: np.ndarray, p: Params) -> np.ndarray:
    """Binary input both line finders consume -- the promoted `stats/hough_lines`
    version, so the study and the registered detector cannot drift apart."""
    return _anomaly_map(field, domain, bin_k=p.bin_k, min_size=p.min_size)


# ==========================================================================
#  RADON -- global straight lines from sinogram peaks
# ==========================================================================
def _radon_geometry(shape: Tuple[int, int]) -> Tuple[Tuple[int, int], int]:
    """Replicate skimage.transform.radon's `circle=False` zero-padding, so a
    sinogram bin can be mapped back to image pixels analytically.

    Returns (pad_before, center): the offset added to (row, col) to reach padded
    coordinates, and the rotation center of the padded square."""
    diagonal = np.sqrt(2) * max(shape)
    pad = [int(np.ceil(diagonal - s)) for s in shape]
    new_center = [(s + p) // 2 for s, p in zip(shape, pad)]
    old_center = [s // 2 for s in shape]
    pad_before = tuple(nc - oc for oc, nc in zip(old_center, new_center))
    return pad_before, (shape[0] + pad[0]) // 2


def projection_coords(shape: Tuple[int, int], angle_deg: float):
    """Rotated-frame coordinates (across, along) of every image pixel, at
    `angle_deg`.

    radon() warps the padded square by R = [[cos, sin, -c(cos+sin-1)],
    [-sin, cos, -c(cos-sin-1)]] (an inverse map in (x, y) = (col, row) order) and
    sums over rows, so sinogram row `r` collects the rotated-frame COLUMN r.
    Inverting that map gives, for a padded pixel (y', x') with
    u = x' + c(cos+sin-1) and v = y' + c(cos-sin-1),
        across = cos*u - sin*v   (the sinogram bin -- constant along the line)
        along  = sin*u + cos*v   (position along the line -- used to trim it)
    Both are returned as full (H, W) arrays."""
    (pr, pc), c = _radon_geometry(shape)
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    u = (np.arange(shape[1]) + pc) + c * (cos_a + sin_a - 1.0)   # per column
    v = (np.arange(shape[0]) + pr) + c * (cos_a - sin_a - 1.0)   # per row
    across = cos_a * u[None, :] - sin_a * v[:, None]
    along = sin_a * u[None, :] + cos_a * v[:, None]
    return across, along


def radon_sinogram(binary: np.ndarray, domain: np.ndarray, p: RadonParams):
    """Support-normalized Radon z-map of the anomaly map `binary` over `domain`.

    The raw sinogram mixes "this chord is full of flagged pixels" with "this
    chord is long / mostly inside the detector". Dividing by the Radon transform
    of the domain indicator turns each bin into the FRACTION of its chord that is
    flagged; bins with too little support are dropped and the rest robust-z'd, so
    the bar is set by the typical chord, not by the detector geometry.

    A z-score alone is not enough to accept a bin: on a run whose residual is
    nearly empty (run 475 leaves ~100 flagged px) the MAD of the fraction map
    collapses and a chord holding three specks scores z >> 10. `min_hits` is the
    absolute floor that keeps the detector honest there -- a line must be made of
    real pixels, not merely of pixels unusual relative to nothing.

    Returns (z, theta, valid): the z-map (n_bins, n_angles), the angles in
    degrees, and the acceptance mask (support AND `min_hits`)."""
    theta = np.linspace(0.0, 180.0, p.n_angles, endpoint=False)
    sino = radon((binary & domain).astype(np.float64), theta=theta,
                 circle=False, preserve_range=True)
    supp = radon(domain.astype(np.float64), theta=theta, circle=False,
                 preserve_range=True)
    valid = supp > p.min_support * supp.max()
    frac = np.zeros_like(sino)
    frac[valid] = sino[valid] / supp[valid]
    # z is computed over the support-valid bins (that is the population it is
    # measured against); acceptance additionally requires the evidence floor.
    return robust_z(frac, valid), theta, valid & (sino >= p.min_hits)


def radon_lines(binary: np.ndarray, domain: np.ndarray, p: RadonParams,
                sino=None):
    """Boolean mask of the straight lines Radon finds in the anomaly map.

    Sinogram peaks above `k` (separated by `min_distance` bins so one thick line
    is not counted many times) are back-projected as bands of half-width `width`.
    With `trim`, a band is clipped along its own direction to the span between
    its first and last flagged pixel (plus `trim_pad`), so a 300-px scratch masks
    300 px instead of the full 1400-px chord it happens to lie on, while still
    bridging the faint stretches in between.

    `sino` is an already-computed `radon_sinogram` triple -- the transform is the
    expensive part and does not depend on `k`/`width`/`trim`, so the sweep reuses
    one. Returns (mask, z, theta, accept, peaks)."""
    z, theta, accept = sino if sino is not None else radon_sinogram(binary, domain, p)
    peaks = peak_local_max(np.where(accept, z, 0.0), min_distance=p.min_distance,
                           threshold_abs=p.k, num_peaks=p.max_lines,
                           exclude_border=False)

    mask = np.zeros(binary.shape, dtype=bool)
    for r, i in peaks:
        across, along = projection_coords(binary.shape, theta[i])
        band = np.abs(across - r) <= p.width
        if p.trim:
            ev = along[band & binary]
            if ev.size == 0:
                continue
            band &= ((along >= ev.min() - p.trim_pad) &
                     (along <= ev.max() + p.trim_pad))
        mask |= band
    return mask & domain, z, theta, accept, peaks


# ==========================================================================
#  HOUGH -- segments from the thresholded darkness field
# ==========================================================================
def hough_lines(binary: np.ndarray, domain: np.ndarray, p: HoughParams):
    """Boolean mask of the Hough SEGMENTS found in the anomaly map.

    `probabilistic_hough_line` returns endpoint pairs; each is rasterized and
    dilated by `width`. Segments are inherently bounded, so no trimming step is
    needed here. Both steps come from the promoted `stats/hough_lines`; the study
    keeps the segment list around because the figure draws it. Returns
    (mask, segments)."""
    segments = hough_segments(binary, threshold=p.threshold,
                              line_length=p.line_length, line_gap=p.line_gap)
    return segments_mask(segments, binary.shape, width=p.width) & domain, segments


# ==========================================================================
#  per-run study
# ==========================================================================
def context(run: int, p: Params):
    """Everything a line finder needs for `run`: the sample, the production mask
    it must improve on, the residual domain, the darkness field and the shared
    anomaly map. Shared by `study_run` and `sweep_run`."""
    # line_detector=False: the baseline is production WITHOUT the line detector
    # this study exists to justify -- otherwise `prod` already contains the Hough
    # pixels and every "added" column reads zero.
    pipe = production_pipeline("union", line_detector=False)
    sample = load_sample(
        run, selection=pipe.shot_selection, reductions=pipe.reductions_needed(),
        calibrations=pipe.calibrations_needed(),
    )
    prod = pipe.run(sample)                 # per-pixel production baseline
    domain = sample.real & ~prod            # only NEW pixels count
    field = darkness_field(sample, radius=p.blackhat_radius)
    return sample, prod, domain, field, anomaly_map(field, domain, p)


def _added_scores(M, prod, human):
    """(full-mask scores, added %, precision of the added pixels)."""
    added = M & ~prod
    ap = float((added & human).sum() / max(added.sum(), 1))
    return score(M, human), 100 * float(added.mean()), ap


def study_run(run: int, p: Params = None):
    """Run both detectors on `run`, print the score table, save the figure."""
    p = p or Params()
    sample, prod, domain, field, binary = context(run, p)
    rad, zsino, theta, accept, peaks = radon_lines(binary, domain, p.radon)
    hgh, segments = hough_lines(binary, domain, p.hough)

    variants = {
        "production": prod,
        "prod+radon": prod | rad,
        "prod+hough": prod | hgh,
        "prod+both": prod | rad | hgh,
    }
    print(f"\n=== run {run} — line detection on the production residual ===")
    print(f"  anomaly map: {int(binary.sum())} px flagged at z>{p.bin_k} "
          f"({100*binary.mean():.3f}% of the chip)")
    print(f"  radon: {len(peaks)} sinogram peaks (k={p.radon.k}), "
          f"{100*rad.mean():.3f}% added")
    print(f"  hough: {len(segments)} segments (len>={p.hough.line_length}), "
          f"{100*hgh.mean():.3f}% added")
    print(f"  {'mask':12s} {'masked%':>8s} {'IoU':>7s} {'prec':>7s} {'rec':>7s} "
          f"{'added%':>8s} {'add-prec':>9s}")
    rows = {}
    for name, M in variants.items():
        s, add, ap = _added_scores(M, prod, sample.human)
        rows[name] = (s, add, ap)
        print(f"  {name:12s} {100*M.mean():8.3f} {s['iou']:7.3f} "
              f"{s['precision']:7.3f} {s['recall']:7.3f} {add:8.3f} {ap:9.3f}")

    out = _figure(run, sample, p, prod, field, domain,
                  rad, zsino, theta, accept, peaks, hgh, binary, segments, rows)
    print(f"  [figure] {out}")
    return rows


# ==========================================================================
#  sensitivity sweep -- are the defaults a plateau or a lucky point?
# ==========================================================================
RADON_GRID = [(k, w) for k in (8.0, 14.0, 25.0, 40.0) for w in (1.0, 2.0, 3.0)]
HOUGH_GRID = [(L, t) for L in (20, 40, 100, 200) for t in (5, 10, 20)]


def sweep_run(run: int, p: Params = None):
    """Walk each detector's two main knobs on one run and print IoU / added% /
    added-precision, so the defaults can be read off a plateau instead of a
    single lucky point. The Radon sinogram (the expensive part) is computed once
    and reused across the grid; only peak-picking and back-projection repeat."""
    p = p or Params()
    sample, prod, domain, field, binary = context(run, p)
    base = score(prod, sample.human)["iou"]
    print(f"\n=== run {run} — sensitivity sweep (production IoU {base:.3f}) ===")

    sino = radon_sinogram(binary, domain, p.radon)
    print(f"  RADON   {'k':>5s} {'width':>6s} {'lines':>6s} {'added%':>8s} "
          f"{'IoU':>7s} {'dIoU':>7s} {'add-prec':>9s}")
    for k, w in RADON_GRID:
        rp = replace(p.radon, k=k, width=w)
        M, _, _, _, peaks = radon_lines(binary, domain, rp, sino=sino)
        s, add, ap = _added_scores(prod | M, prod, sample.human)
        print(f"  {'':7s} {k:5.1f} {w:6.1f} {len(peaks):6d} {add:8.3f} "
              f"{s['iou']:7.3f} {s['iou']-base:+7.3f} {ap:9.3f}")

    print(f"  HOUGH   {'len':>5s} {'votes':>6s} {'segs':>6s} {'added%':>8s} "
          f"{'IoU':>7s} {'dIoU':>7s} {'add-prec':>9s}")
    for L, t in HOUGH_GRID:
        hp = replace(p.hough, line_length=L, threshold=t)
        M, segments = hough_lines(binary, domain, hp)
        s, add, ap = _added_scores(prod | M, prod, sample.human)
        print(f"  {'':7s} {L:5d} {t:6d} {len(segments):6d} {add:8.3f} "
              f"{s['iou']:7.3f} {s['iou']-base:+7.3f} {ap:9.3f}")


def _base_display(sample):
    """arcsinh-compressed sum image + robust display limits (as in the frangi study)."""
    base = np.arcsinh(sample.sumimg /
                      (np.nanmedian(np.abs(sample.sumimg[sample.real])) + 1e-9))
    vlo, vhi = np.nanpercentile(base[sample.real], [2, 99])
    return base, vlo, vhi


def _overlay(ax, base, vlo, vhi, prod, added, title):
    """Sum image, production mask in blue, the detector's new pixels in red."""
    ax.imshow(base, cmap="gray", vmin=vlo, vmax=vhi)
    blue = np.zeros((*prod.shape, 4)); blue[prod] = (0.1, 0.4, 1.0, 0.40)
    red = np.zeros((*prod.shape, 4)); red[added] = (1.0, 0.0, 0.0, 1.0)
    ax.imshow(blue); ax.imshow(red)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _figure(run, sample, p, prod, field, domain,
            rad, zsino, theta, accept, peaks, hgh, binary, segments, rows):
    base, vlo, vhi = _base_display(sample)
    fig, axes = plt.subplots(2, 4, figsize=(21, 11))
    A = axes.ravel()

    # 1. what production already masks
    sp = rows["production"][0]
    A[0].imshow(base, cmap="gray", vmin=vlo, vmax=vhi)
    blue = np.zeros((*prod.shape, 4)); blue[prod] = (0.1, 0.4, 1.0, 0.45)
    A[0].imshow(blue)
    A[0].set_title(f"production mask (blue) — {100*prod.mean():.2f}% masked\n"
                   f"IoU {sp['iou']:.3f}  p {sp['precision']:.2f}  "
                   f"r {sp['recall']:.2f}", fontsize=9)
    A[0].set_xticks([]); A[0].set_yticks([])

    # 2. the shared evidence field, on the residual domain only
    shown = np.where(domain, field, np.nan)
    A[1].imshow(shown, cmap="magma", vmin=0,
                vmax=np.nanpercentile(shown, 99.5) if domain.any() else 1)
    A[1].set_title(f"black-hat darkness z (radius {p.blackhat_radius})\n"
                   "on residual domain = real & ~production", fontsize=9)
    A[1].set_xticks([]); A[1].set_yticks([])

    # 3. the sinogram and its accepted peaks
    # bins failing the support / min_hits gate are shown as NaN (grey), so a run
    # that abstains reads as "nothing was eligible", not "nothing was bright".
    A[2].imshow(np.where(accept, zsino, np.nan), aspect="auto", cmap="viridis",
                origin="lower", extent=[theta[0], theta[-1], 0, zsino.shape[0]],
                vmin=-3, vmax=max(p.radon.k * 1.5, 6))
    if len(peaks):
        A[2].scatter(theta[peaks[:, 1]], peaks[:, 0], s=40, facecolors="none",
                     edgecolors="red", linewidths=1.2)
    A[2].set_facecolor("0.85")
    A[2].set_xlabel("angle (deg)"); A[2].set_ylabel("offset bin")
    A[2].set_title(f"Radon: sinogram z on bins passing support + "
                   f"min_hits={p.radon.min_hits}\n"
                   f"{len(peaks)} peaks above k={p.radon.k} "
                   f"({100*accept.mean():.0f}% of bins eligible)", fontsize=9)

    # 4. what Radon would add
    s, add, ap = rows["prod+radon"]
    _overlay(A[3], base, vlo, vhi, prod, rad,
             f"+ Radon lines (red, half-width {p.radon.width:g}px)\n"
             f"+{add:.3f}%  IoU {s['iou']:.3f}  added-prec {ap:.2f}")

    # 5. the binarized field Hough consumes, with its segments
    A[4].imshow(binary, cmap="gray_r")
    for (x0, y0), (x1, y1) in segments:
        A[4].plot([x0, x1], [y0, y1], "-", color="red", lw=1.0)
    A[4].set_title(f"shared anomaly map: darkness z > {p.bin_k}, "
                   f"specks < {p.min_size}px dropped\n"
                   f"Hough: {len(segments)} segments, len>={p.hough.line_length}px",
                   fontsize=9)
    A[4].set_xticks([]); A[4].set_yticks([])

    # 6. what Hough would add
    s, add, ap = rows["prod+hough"]
    _overlay(A[5], base, vlo, vhi, prod, hgh,
             f"+ Hough segments (red, dilated {p.hough.width}px)\n"
             f"+{add:.3f}%  IoU {s['iou']:.3f}  added-prec {ap:.2f}")

    # 7. the mask the pipeline would produce with the new detector in it
    s, add, ap = rows["prod+both"]
    _overlay(A[6], base, vlo, vhi, prod, rad | hgh,
             f"PIPELINE + line detector (union)\n"
             f"{100*(prod | rad | hgh).mean():.2f}% masked  IoU {s['iou']:.3f}  "
             f"p {s['precision']:.2f}  r {s['recall']:.2f}")

    # 8. agreement of that mask against the human reference
    full = prod | rad | hgh
    A[7].imshow(agree_rgb(full, sample.human))
    A[7].set_title("agreement vs human reference\n"
                   "green=TP  red=FP  blue=FN", fontsize=9)
    A[7].set_xticks([]); A[7].set_yticks([])

    fig.suptitle(f"run {run} — Radon / Hough line detection added to the "
                 f"production pipeline", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, f"line_detection_run{run:04d}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# ==========================================================================
#  self-test -- the back-projection geometry must match skimage's radon
# ==========================================================================
def selftest(shape=(151, 121), tol=1.5) -> None:
    """Draw known lines, radon them, and check that the peak bin back-projects
    onto the same pixels. Guards `projection_coords` against a skimage change in
    radon's padding/rotation convention."""
    for angle in (0.0, 27.0, 63.0, 115.0, 170.0):
        img = np.zeros(shape)
        rr, cc = draw_line(int(0.2 * shape[0]), int(0.15 * shape[1]),
                           int(0.85 * shape[0]), int(0.8 * shape[1]))
        img[rr, cc] = 1.0
        theta = np.array([angle])
        sino = radon(img, theta=theta, circle=False, preserve_range=True)
        r = int(np.argmax(sino[:, 0]))
        across, _ = projection_coords(shape, angle)
        hit = np.abs(across[rr, cc] - r).min()
        # the drawn line's own pixels must project into (or next to) the peak bin
        assert hit <= tol, f"angle {angle}: nearest projection {hit:.2f} bins off"
        print(f"  angle {angle:6.1f} deg: peak bin {r:4d}, "
              f"line pixels project within {hit:.2f} bins  OK")
    print("selftest passed")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--sweep" in sys.argv:
        for run in EVAL_RUNS:
            sweep_run(run)
        return
    for run in EVAL_RUNS:
        study_run(run)


if __name__ == "__main__":
    main()
