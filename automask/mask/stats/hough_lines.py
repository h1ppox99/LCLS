"""
stats/hough_lines.py -- straight-line defect detection by probabilistic Hough.

Every other statistic in this package is per-pixel: it scores each pixel from its
own value and its neighbourhood, and knows nothing about shape. This one is a
SHAPE detector. It exists because the defects the per-pixel detectors miss most
visibly are extended straight structures -- wire/edge shadows, scratches, dark
seams -- whose individual pixels sit too close to the bulk to clear a robust-z
bar but which are unmistakable as a line.

Three steps:

  1. darkness evidence -- a black-hat z-score of the run-sum image (high == darker
     than the local surroundings), because line defects on this detector are dark.
     The excluded region (dead pixels, and the floor when hidden) is first
     inpainted to its nearest live value: a black-hat over a flat fill leaves an
     intensity step at a masked band edge and turns it into a bright straight RIM
     just outside it, which Hough would connect as a false line. A nearest-value
     fill removes the step, so the strongest straight responses left are the
     genuine lines (see `darkness_z`).
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

`exclude_floor` (default True) hides the geometry+calib floor from step 2. Those
regions are already masked and are the darkest, straightest things on the chip;
left in, they dominate the accumulator and Hough spends its segments
re-discovering the module gap. The stat cannot see the other detectors' picks
(it runs beside them), so the floor is the only exclusion available -- and the
one that matters.

Dropping the floor pixels from the accumulator is necessary but not sufficient:
the black-hat rim sits just OUTSIDE the floor and survives the `~floor` gate. It
is killed at the source by the step-1 inpaint, not by dilating a margin off the
floor -- so genuine lines that run close to a bad region are kept, not sacrificed.
Before the inpaint the rim was 100% of Hough's output on the local runs; after it,
the run-389 dark diagonal shadows are what Hough connects.

Tuned on runs 389/475. With the rim gone the real lines are faint and broken, so
connection is loose (`line_length` 40, `line_gap` 25, `bin_k` 3.5, de-speck 2): on
389 this traces the dark diagonal shadows and on 475 -- which has no such lines --
it adds nothing. `width=1` gives the best precision on the added pixels.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line
from skimage.morphology import binary_dilation, black_tophat, disk, remove_small_objects
from skimage.transform import probabilistic_hough_line

from automask.mask.stats.base import Panel, StatSpec, register_stat, robust_z


@dataclass
class HoughLinesParams:
    radius: int = 5  # black-hat structuring element, px
    bin_k: float = 3.5  # robust-MAD z bar binarizing the darkness field
    min_size: int = 2  # drop connected specks below this many px
    threshold: int = 10  # Hough accumulator votes needed
    line_length: int = 40  # shortest segment kept, px
    line_gap: int = 25  # gap tolerated inside one segment, px
    width: int = 1  # dilation radius applied to rasterized segments, px
    exclude_floor: bool = True  # hide geometry+calib from the accumulator


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
    return probabilistic_hough_line(
        binary, threshold=threshold, line_length=line_length, line_gap=line_gap, rng=rng
    )


def floor_mask(sample, pad=2):
    """The geometry+status floor, recomputed from `sample` alone. Duplicates what
    Pipeline.floor does, because a stat runs beside the floor stats and is not
    handed their output."""
    from automask.mask.stats.geometry import geometry_mask
    from automask.mask.stats.status_as_mask import compute as status_mask

    return geometry_mask(sample.real, pad=pad) | status_mask(sample)


def darkness_z(image, real, floor=None, radius=5):
    """Black-hat darkness z-score with the excluded region inpainted first.

    `live` is `real`, minus `floor` when given. Every non-live pixel is filled with
    its nearest live value BEFORE `black_tophat`, so a masked band carries no
    intensity step for the morphology to raise into a bright edge rim (a false
    straight line Hough would connect); z is scaled over `live`."""
    live = real if floor is None else (real & ~floor)
    idx = ndi.distance_transform_edt(~live, return_indices=True, return_distances=False)
    R = black_tophat(image[tuple(idx)], footprint=disk(radius))
    return robust_z(R, live)


def hough_lines_mask(image, real, p: HoughLinesParams, floor=None):
    """Boolean mask of the Hough segment pixels. Split out from `compute` so the
    study can drive it with an arbitrary domain."""
    domain = real if floor is None else (real & ~floor)
    z = darkness_z(image, real, floor, radius=p.radius)
    binary = anomaly_map(z, domain, bin_k=p.bin_k, min_size=p.min_size)
    segments = hough_segments(
        binary, threshold=p.threshold, line_length=p.line_length, line_gap=p.line_gap
    )
    return segments_mask(segments, binary.shape, width=p.width) & domain


def compute(sample, params: HoughLinesParams | None = None):
    p = params or HoughLinesParams()
    floor = floor_mask(sample) if p.exclude_floor else None
    return hough_lines_mask(sample.mean, sample.real, p, floor=floor)


def explain(sample, params: HoughLinesParams | None = None) -> dict:
    """The three stages a pick hides behind its one boolean output: the graded
    darkness field with its `bin_k` binarizing cut, the binary map Hough votes
    on, and the segments it returned. Mirrors `hough_lines_mask` step for step."""
    p = params or HoughLinesParams()
    floor = floor_mask(sample) if p.exclude_floor else None
    domain = sample.real if floor is None else (sample.real & ~floor)
    z = darkness_z(sample.mean, sample.real, floor, radius=p.radius)
    binary = anomaly_map(z, domain, bin_k=p.bin_k, min_size=p.min_size)
    segments = hough_segments(
        binary, threshold=p.threshold, line_length=p.line_length, line_gap=p.line_gap
    )
    seg = segments_mask(segments, binary.shape, width=p.width) & domain
    return {
        "darkness z": Panel(z, threshold=(p.bin_k, "high")),
        "binary input": Panel(binary, mask=True),
        "segments": Panel(seg, mask=True),
    }


register_stat(
    StatSpec(
        name="hough_lines",
        compute=compute,
        params=HoughLinesParams,
        kind="pick",
        needs=("mean", "real", "status_as_mask"),
        explain=explain,
        doc="probabilistic-Hough segments of the black-hat darkness map; "
        "emits a mask (kind='pick'), so field_reg must be None",
    )
)
