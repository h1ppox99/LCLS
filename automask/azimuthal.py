#!/usr/bin/env python3
"""
azimuthal.py -- pyFAI 1-D integration of the assembled Jungfrau1M images,
with the masks the production pipeline produces.

The point is to see what masking *does to the physics*: a mask is only worth
anything if the I(q) it yields is clean. So this runs
`masking.production_pipeline()` on a run and integrates the run's sum image
with the mask it produces.

Geometry follows CLAUDE.md: sample-detector 190 mm (NOT the stale 100 mm psana
z map), lambda = 1.2915 A, 75 um pixels. The beam center comes from
`geometry.get_center`, already in the (axis0, axis1) order of the frozen
assembled arrays; pyFAI's poni1/poni2 are (slow, fast) = (axis0, axis1) in
metres, so the mapping is a straight multiply by the pixel size.

Two mask conventions meet here and they happen to agree: this project's is
bool with True == masked, pyFAI's is "nonzero == ignored".

Panel placement comes directly from psana. The optical constants below remain
experiment-specific because the stored psana z map has the wrong distance.

Canvas pixels no panel maps onto (the assembly gaps) are never real data, so
they are folded into the mask regardless of what the pipeline says about them.

Run:  python -m automask.azimuthal             # both EVAL_RUNS
      python -m automask.azimuthal 475
"""
from __future__ import annotations
import os
from typing import Optional

import numpy as np

from automask import geometry
from automask.dataset import load_image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")

DIST_M = 0.190          # sample -> detector
WAVELENGTH_M = 1.2915e-10
PIXEL_M = 75e-6
NPT = 256
MIN_COUNT = 200         # bins thinner than this are noise, not signal

# pyFAI's polarization_factor is signed by which detector axis the E-field lies
# along: +1 = axis2 (fast), -1 = axis1 (slow). Our assembled arrays are
# transposed (axis0 = ix = lab x), so the horizontal LCLS polarization is -1
# here. Verified, not assumed: at -1 `ai.polarization()` reproduces psana's own
# `azav__azav_pol` array to 1e-4 over all 1M pixels, +1 anti-correlates.
POLARIZATION = -1.0


def dead_canvas(run: int) -> np.ndarray:
    """Assembled pixels that no detector pixel maps onto (True == not data)."""
    ix, iy = geometry.index_maps(run)
    covered = np.zeros(geometry.asm_shape(run), dtype=bool)
    covered[ix, iy] = True
    return ~covered


def integrator(run: int, shape: Optional[tuple] = None):
    """AzimuthalIntegrator for `run`'s assembled canvas, in q [A^-1]."""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    shape = shape or geometry.asm_shape(run)
    c0, c1 = geometry.get_center(run)
    det = Detector(pixel1=PIXEL_M, pixel2=PIXEL_M, max_shape=shape)
    # PONI is measured from the CORNER of pixel [0,0], while `get_center`
    # returns a pixel INDEX, whose centre sits half a pixel further out. Skipping
    # the +0.5 is a 37.5 um offset: small, but it dominates the residual --
    # against psana's per-pixel q map it is the difference between an rms of
    # 4.9e-4 and 7e-5 A^-1 (the latter being the assembly's own rounding floor).
    return AzimuthalIntegrator(dist=DIST_M, poni1=(c0 + 0.5) * PIXEL_M,
                               poni2=(c1 + 0.5) * PIXEL_M, detector=det,
                               wavelength=WAVELENGTH_M)


def integrate(img, run: int, mask=None, npt: int = NPT, ai=None,
              min_count: int = MIN_COUNT):
    """1-D azimuthal average of `img` ignoring `mask` (True == masked).

    Returns `(q [A^-1], I, sigma)`. Solid-angle, polarization and Poisson
    errors are all on -- an uncorrected profile is wrong by up to 12% in the
    detector corners, which is larger than most of the features being looked at.

    Bins backed by fewer than `min_count` pixels come back as NaN. The frozen
    geometry puts the beam near a corner, so the extreme-q rings clip the
    detector with only tens of pixels; left in, they show up as a spurious
    upturn at the end of the curve that is pure counting noise.
    """
    ai = ai or integrator(run, np.asarray(img).shape)
    m = dead_canvas(run)
    if mask is not None:
        m = m | np.asarray(mask, dtype=bool)
    res = ai.integrate1d(np.asarray(img, dtype=np.float64), npt,
                         mask=m.astype(np.uint8), unit="q_A^-1",
                         correctSolidAngle=True,
                         polarization_factor=POLARIZATION,
                         error_model="poisson")
    thin = res.count < min_count
    return (res.radial, np.where(thin, np.nan, res.intensity),
            np.where(thin, np.nan, res.sigma))


def plot_run(run: int, npt: int = NPT, out: Optional[str] = None):
    """Integrate run `run` unmasked / floor-only / full production mask, plot."""
    from automask.masking import production_pipeline
    from automask.evaluation import load_sample
    import matplotlib.pyplot as plt

    pipe = production_pipeline("union")
    sample = load_sample(
        run, selection=pipe.shot_selection, reductions=pipe.reductions_needed(),
        calibrations=pipe.calibrations_needed(),
    )
    mask = pipe.run(sample)
    img = load_image(f"sum_calib_run{run:04d}").astype(np.float64)

    q, I, sig = integrate(img, run, mask, npt)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(q, I - sig, I + sig, color="tab:blue", alpha=0.25, lw=0)
    ax.plot(q, I, "-", color="tab:blue", lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("q  [Å$^{-1}$]")
    ax.set_ylabel("I(q)  [ADU, solid-angle corrected]")
    ax.set_title(f"run {run} — azimuthal integration, production mask "
                 f"({100*mask.mean():.1f}% of the canvas masked)")
    ax.grid(alpha=0.25)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = out or os.path.join(FIG_DIR, f"azimuthal_run{run:04d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[figure] {out}")

    prof = os.path.join(HERE, "outputs", "masks", f"azav_run{run:04d}.npz")
    np.savez(prof, q=q, production=I, sigma=sig)
    print(f"[profile] {prof}")
    return fig


def main(runs=None):
    from automask.evaluation import EVAL_RUNS
    for run in (EVAL_RUNS if runs is None else runs):
        plot_run(run)


if __name__ == "__main__":
    import sys
    main([int(a) for a in sys.argv[1:]] or None)
