"""
stats/hough_lines.py -- straight-line defect detection by probabilistic Hough.

Every other statistic in this package is per-pixel: it scores each pixel from its
own value and its neighbourhood, and knows nothing about shape. This one is a
SHAPE detector. It exists because the defects the per-pixel detectors miss most
visibly are extended straight structures -- wire/edge shadows, scratches, dark
seams -- whose individual pixels sit too close to the bulk to clear a robust-z
bar but which are unmistakable as a line.

Three steps:

  1. darkness evidence -- the black-hat z-score of the run-sum image (same
     construction as `stats/blackhat.py`, high == darker than the local
     surroundings), because line defects on this detector are dark.
  2. binarize -- z > `bin_k`, de-specked at `min_size`. Hough votes are counted,
     not weighted, so the input must be binary; the de-speck is deliberately
     gentle (a few px) so it removes isolated noise without breaking the sparse,
     broken runs that ARE the faint lines.
  3. `probabilistic_hough_line` -- returns segments with endpoints, which are
     rasterized and dilated by `width`.

This is the registry's first ``kind="pick"`` stat: `compute` returns the boolean
mask directly rather than a graded field. Hough is a decision procedure, not a
measurement -- it votes on SEGMENTS and a pixel is on one or it is not -- so
there is no per-pixel score for a threshold to act on, and the decisive knobs
(`line_length`, `threshold`, `line_gap`) all sit upstream of any pixel. Hence no
`k` / `mode` here, and `Detector` skips its field-regularizer and threshold
stages (setting a `field_reg` on this stat raises; see `Detector.__post_init__`).

`defectiveness_scale` is the opt-in for the fusion combiners (`weighted_sum`,
`mahalanobis`), which sum robust-z fields a pick has no equivalent of. It is the
z-value one picked pixel is worth; leave it None and fusion raises rather than
silently contributing a value below the combiner's threshold. To let this
detector mask on its own under `weighted_sum` (k=3.5), set it above that.

`exclude_floor` (default True) hides the geometry+calib floor from step 2. Those
regions are already masked and are the darkest, straightest things on the chip;
left in, they dominate the accumulator and Hough spends its segments
re-discovering the module gap. The stat cannot see the other detectors' picks
(it runs beside them), so the floor is the only exclusion available -- and the
one that matters.

Tuned on runs 389/475 in `studies/line_detection.py`, which also measures the
Radon alternative. `line_length` sits on a broad plateau (100-200 px) and the
vote threshold barely matters; `width=1` gives the best precision on the added
pixels at equal IoU.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from skimage.draw import line as draw_line
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.transform import probabilistic_hough_line

from automask.stats.base import StatSpec, register_stat
from automask.stats.blackhat import blackhat_stat


@dataclass
class HoughLinesParams:
    radius: int = 5          # black-hat structuring element, px
    bin_k: float = 4.0       # robust-MAD z bar binarizing the darkness field
    min_size: int = 4        # drop connected specks below this many px
    threshold: int = 10      # Hough accumulator votes needed
    line_length: int = 100   # shortest segment kept, px
    line_gap: int = 5        # gap tolerated inside one segment, px
    width: int = 1           # dilation radius applied to rasterized segments, px
    exclude_floor: bool = True   # hide geometry+calib from the accumulator
    # z-worth of one picked pixel, for consumes="fields" combiners only.
    # None (default) => fusing this detector is an error rather than a silent no-op.
    defectiveness_scale: float | None = None


def anomaly_map(field, domain, bin_k=4.0, min_size=4):
    """Binary line-finder input: darkness z above `bin_k` inside `domain`, with
    connected components smaller than `min_size` dropped."""
    b = (field > bin_k) & domain
    return remove_small_objects(b, min_size=min_size) if min_size > 1 else b


def segments_mask(segments, shape, width=1):
    """Rasterize `probabilistic_hough_line` endpoint pairs into a boolean mask,
    dilated by `width`. Hough returns its points in (col, row) order."""
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for (x0, y0), (x1, y1) in segments:
        rr, cc = draw_line(int(y0), int(x0), int(y1), int(x1))
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        mask[rr[ok], cc[ok]] = True
    return binary_dilation(mask, disk(width)) if width > 0 else mask


def hough_segments(binary, threshold=10, line_length=100, line_gap=5, rng=0):
    """Probabilistic Hough segments of a binary map. `rng` is pinned so the same
    input gives the same segments -- the transform samples edge points."""
    return probabilistic_hough_line(binary, threshold=threshold,
                                    line_length=line_length, line_gap=line_gap,
                                    rng=rng)


def floor_mask(sample, pad=2):
    """The geometry+calib floor, recomputed from `sample` alone. Duplicates what
    Pipeline.floor does, because a stat runs beside the floor stats and is not
    handed their output."""
    from automask.stats.geometry import geometry_mask
    return geometry_mask(sample.real, pad=pad) | sample.calib


def hough_lines_mask(sumimg, real, p: HoughLinesParams, floor=None):
    """Boolean mask of the Hough segment pixels. Split out from `compute` so the
    study can drive it with an arbitrary domain."""
    domain = real if floor is None else (real & ~floor)
    z = blackhat_stat(sumimg, real, radius=p.radius)
    binary = anomaly_map(z, domain, bin_k=p.bin_k, min_size=p.min_size)
    segments = hough_segments(binary, threshold=p.threshold,
                              line_length=p.line_length, line_gap=p.line_gap)
    return segments_mask(segments, binary.shape, width=p.width) & domain


def compute(sample, params: HoughLinesParams | None = None):
    p = params or HoughLinesParams()
    floor = floor_mask(sample) if p.exclude_floor else None
    return hough_lines_mask(sample.sumimg, sample.real, p, floor=floor)


register_stat(StatSpec(
    name="hough_lines",
    compute=compute,
    params=HoughLinesParams,
    kind="pick",
    needs=("sumimg", "real", "calib"),
    doc="probabilistic-Hough segments of the black-hat darkness map; "
        "emits a mask (kind='pick'), so field_reg must be None",
))
