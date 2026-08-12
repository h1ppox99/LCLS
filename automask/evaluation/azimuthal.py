"""
Azimuthal physical-consistency diagnostics for runtime evaluation.

The first hypothesis in this package that is about the EXPERIMENT rather than
about estimation. The sample scatters isotropically, so after solid-angle and
polarization correction every pixel at the same |q| measures the same quantity:

    I_i = mu(q_i) + eps_i,      eps_i ~ N(0, sigma(q_i)^2)

A correct mask leaves a ring whose azimuthal sectors all estimate one mu(q); an
unmasked defect makes one sector disagree by more than sigma(q)/sqrt(n). This is
the closest thing to a ground truth the production setting offers, because it
scores the mask by what the mask is FOR.

WHY NOT A SIGNIFICANCE TEST. One-way ANOVA across sectors (Welch's F) is the
obvious construction and it is the wrong one, measurably: with ~200 px per cell
the noise on a sector mean is ~0.1% while the real anisotropy of the sample
(Bragg spots, texture) is percent-level, so H0 is false everywhere and F runs to
100-250 whatever the mask does. Worse, F goes UP as the mask improves -- removing
defect pixels shrinks the within-cell variance in its denominator, making the
surviving true anisotropy more significant. A significance test answers "is the
ring inhomogeneous?" (always yes) when the question is "by how much, and did the
mask reduce it?".

THE STATISTIC is an effect size in units of the ring's own intensity. Per ring:

    V_b    = Var_s(m_s)                    observed scatter of sector means
    E[V_b] = sigma_hat^2 * mean_s(1/n_s)   the part noise explains under H0
    excess = sqrt(max(V_b - E[V_b], 0)) / |mu|

Scale-free, insensitive to n, and it does not inflate as the data get cleaner.

`sigma_hat^2` AND `mu` ARE FROZEN, not read off the candidate's own survivors.
Estimating them per candidate leaves a milder form of the same defect that sinks
Welch's F: a mask that removes noisy pixels shrinks `sigma_hat^2`, shrinking the
term SUBTRACTED from its own numerator, so improving the mask inflates the
statistic; and a mask that removes hot pixels lowers `mu`, inflating it again
through the divisor. Neither term is a property of the mask -- both describe the
ring -- so `ring_reference` computes them once from the geometry+calib floor
pixel set, which every candidate shares by construction. What remains
mask-dependent is `V_b` and the cell counts, which is the only part that should
carry a ranking. Freezing the pixel SET is not enough on its own -- the floor
still contains the intensity defects, so `sigma_hat^2` has to be reduced over it
robustly or it inherits them and the subtracted floor swallows the signal; see
`ring_reference`.

ITS NULL IS NOT ZERO. `V_b` is itself estimated from only S sector means, so it
scatters around its expectation and the clip at zero makes the residual
one-sided. Measured on synthetic isotropic rings (12 sectors, 200 px/cell, in
`tests/test_evaluation_runtime.py`) that sampling floor is ~0.19% of the ring mean,
against a raw sector-mean scatter of 0.7% -- the subtraction removes about three
quarters of the noise, and the rest is the metric's resolution limit. A 5%
one-sector anomaly reads 1.5%, so the working dynamic range at this cell size is
roughly a factor of eight. Differences below the floor are not interpretable,
which is why the control terms below matter more than the raw value.

THE CONTROL is what makes it a metric rather than a tautology. Masking can only
remove information, and dropping extreme pixels always flatters any homogeneity
measure -- so every candidate is paired with a RANDOM mask of the same size drawn
from the same pixels. `azim_gain` (control minus candidate) and `azim_winrate`
(paired per-ring sign test against the control) are the honest quantities;
`azim_excess` alone rewards masking more.

TWO SYSTEMATICS, quantified rather than assumed:
  * Bragg/texture anisotropy is real physics, not a mask failure. Suppressed by
    per-ring winsorization at the 99th percentile, which leaves broad defects
    untouched.
  * Radial-gradient leakage: sectors do not share a mean q (the beam sits near a
    corner), so |dI/dq| * spread_s(q_s) turns a radial slope into fake azimuthal
    scatter. `gradient_leakage` computes it per ring -- it is the floor below
    which an `excess` value means nothing.

The public diagnostic uses repeated size-matched random controls. This module
contains the numerical primitives and does not register global metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_MIN = 30          # minimum pixels for a (ring, sector) cell to be usable
MIN_SECTORS = 4     # a ring needs this many usable sectors to be scored
TARGET_CELL = 200   # target pixels per cell -> sets the number of radial rings
CLIP_Q = 0.99       # per-ring winsorization point (Bragg suppression)
N_SECTORS = 12


# ==========================================================================
#  per-pixel frame
# ==========================================================================
def pixel_frame(sample):
    """Per-pixel (q, chi, corrected intensity, validity) for a Sample's image.

    Reads `sample.mean` so the frame can also be built for a resampled Sample
    (see `evaluation.resampling`).
    """
    from automask import azimuthal as az

    img = np.asarray(sample.mean, dtype=np.float64)
    ai = az.integrator(sample.run, img.shape)
    q = ai.array_from_unit(img.shape, "center", "q_A^-1", scale=True)
    chi = ai.chiArray(img.shape)
    corr = ai.solidAngleArray(img.shape) * ai.polarization(img.shape,
                                                           factor=az.POLARIZATION)
    valid = ~az.dead_canvas(sample.run) & (img != 0) & np.isfinite(corr) & (corr > 0)
    inten = np.where(valid, img / np.where(corr > 0, corr, 1.0), np.nan)
    return q, chi, inten, valid


def ring_edges(q, usable, n_rings):
    """Equal-COUNT radial bin edges.

    Equal-count rather than equal-width: the beam sits near a detector corner, so
    a fixed dq gives rings whose pixel counts span orders of magnitude, and the
    test's power would vary with q for reasons unrelated to the mask. Edges come
    from the UNMASKED pixel set, once, so all candidates are scored on identical
    rings.
    """
    v = np.sort(q[usable])
    idx = np.linspace(0, v.size - 1, n_rings + 1).astype(int)
    return np.unique(v[idx])


def sector_edges_per_ring(ring_idx, chi, keep, n_rings, n_sectors):
    """Per-ring equal-count sector boundaries in chi, as an index array.

    Full-2pi sectors are unusable here: with the beam near a corner only a
    fraction of each ring lies on the detector, and fixed 2pi/S sectors leave
    most cells empty. Each ring's OWN chi coverage is split into S equal-count
    sectors instead, with boundaries derived once from `keep` (the floor-only
    pixel set) so every candidate is compared on the same detector regions.
    """
    sel = np.flatnonzero(keep.ravel())
    r = ring_idx.ravel()[sel]
    c = chi.ravel()[sel]
    order = np.lexsort((c, r))
    sel, r = sel[order], r[order]
    counts = np.bincount(r, minlength=n_rings)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    rank = np.arange(sel.size) - offsets[r]
    sector = (rank * n_sectors // np.maximum(counts[r], 1)).astype(np.int32)
    sector = np.clip(sector, 0, n_sectors - 1)
    out = np.full(chi.size, -1, dtype=np.int32)
    out[sel] = sector
    return out.reshape(chi.shape)


def winsorize_per_ring(inten, ring_idx, usable, n_rings, qcut=CLIP_Q):
    """Clip each ring's intensities at its own `qcut` quantile.

    Bragg spots are real physics, not mask failures, but they dominate any
    variance-based statistic. Clipping the top 1% per ring suppresses them while
    leaving broad, low-contrast defects (the ones the mask is judged on) intact.
    """
    sel = np.flatnonzero(usable.ravel() & np.isfinite(inten.ravel()))
    r = ring_idx.ravel()[sel]
    v = inten.ravel()[sel]
    order = np.lexsort((v, r))
    sel_s, r_s, v_s = sel[order], r[order], v[order]
    counts = np.bincount(r_s, minlength=n_rings)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    thr = np.full(n_rings, np.inf)
    nz = counts > 0
    pick = offsets[:-1][nz] + np.minimum((qcut * counts[nz]).astype(int),
                                         counts[nz] - 1)
    thr[nz] = v_s[pick]
    out = inten.copy()
    flat = out.ravel()
    flat[sel] = np.minimum(flat[sel], thr[ring_idx.ravel()[sel]])
    return out


# ==========================================================================
#  statistics
# ==========================================================================
def cell_moments(ring_idx, sec_idx, inten, keep, n_rings, n_sectors):
    """Per-(ring, sector) count, sum and sum-of-squares over the kept pixels."""
    sel = keep & np.isfinite(inten) & (sec_idx >= 0)
    cell = ring_idx[sel] * n_sectors + sec_idx[sel]
    v = inten[sel]
    size = n_rings * n_sectors
    n = np.bincount(cell, minlength=size).reshape(n_rings, n_sectors)
    s1 = np.bincount(cell, weights=v, minlength=size).reshape(n_rings, n_sectors)
    s2 = np.bincount(cell, weights=v ** 2, minlength=size).reshape(n_rings, n_sectors)
    return n, s1, s2


def _cell_stats(n, s1, s2):
    """Per-(ring, sector) usability, mean and unbiased variance."""
    ok = n >= N_MIN
    m = np.where(ok, s1 / np.maximum(n, 1), np.nan)
    v = np.where(ok, (s2 - n * np.nan_to_num(m) ** 2) / np.maximum(n - 1, 1), np.nan)
    v = np.where(ok & (v > 0), v, np.nan)
    return ok & np.isfinite(v) & np.isfinite(m), m, v


def _ring_mean(ok, m, S):
    return np.where(S > 0, np.nansum(np.where(ok, m, 0), axis=1) / np.maximum(S, 1),
                    np.nan)


def ring_reference(n, s1, s2):
    """Per-ring `(sigma^2, mu)` from a pixel set that no candidate can move.

    Everything in `excess` except the sector scatter itself is a property of the
    RING -- the per-pixel noise scale that sets the subtracted floor, and the
    intensity that makes the result a fraction. Estimating either from the
    candidate's own surviving pixels couples them to the mask in the wrong
    direction (see `excess_scatter`), so both are frozen once on the floor-only
    pixels, which are identical for every candidate by construction.

    `sigma^2` is the MEDIAN of the per-cell variances across the ring's sectors,
    not the dof-weighted pool the candidate-local form used. The floor removes
    geometry and calib defects but not the intensity defects the mask exists to
    find, so those pixels are still in this set, and pooling lets a defect
    confined to one or two sectors set the noise scale for the whole ring.
    Measured on a synthetic ring stack with one sector at 8x the per-pixel
    variance, the pooled reduction returns 644 against a true 100 and the
    subtracted floor then swallows a real 4% anisotropy whole -- every candidate
    reads exactly 0.000% and the metric stops discriminating (67% of rings clip).
    The median returns 100.2 on the same field. Sectors are the natural axis for
    this: a defect that spans all of them uniformly is not something the
    azimuthal test can see anyway. On a ring with no variance-inflating defect
    the two agree to three decimals, so this does not move the clean case.
    """
    ok, m, v = _cell_stats(n, s1, s2)
    S = ok.sum(axis=1)
    sig2 = np.nanmedian(np.where(ok, v, np.nan), axis=1)
    return np.where(S > 0, sig2, np.nan), _ring_mean(ok, m, S)


def excess_scatter(n, s1, s2, ref=None):
    """Effect size: azimuthal scatter of sector means beyond the noise floor.

    Returns (excess, mu, welch_F) with `excess` a FRACTION of the ring mean.
    Welch's F is carried only to document why a significance test misleads here
    (see the module docstring).

    `ref` is the frozen `(sigma^2, mu)` from `ring_reference`. With it, the only
    mask-dependent quantities left in `excess` are the sector scatter `Vb` and
    the cell counts that set its expectation -- so a difference between two
    candidates is a difference in azimuthal consistency, not in what they left
    behind to estimate the noise with. Passing `ref=None` reproduces the
    self-referential form and is kept for the sector-count study, which needs a
    reference per partition.

    `Vb` stays centred on the candidate's OWN ring mean even when `ref` is given:
    it is a variance about a mean, and centring it on a foreign one would fold
    `(mu_cand - mu_ref)^2` -- a radial offset, not azimuthal scatter -- into the
    numerator. The frozen `mu` is the unit, not the centre.
    """
    ok, m, v = _cell_stats(n, s1, s2)
    nn = np.where(ok, n, 0).astype(np.float64)
    S = ok.sum(axis=1)

    mu = _ring_mean(ok, m, S)
    Vb = np.where(S > 1, np.nansum(np.where(ok, (m - mu[:, None]) ** 2, 0), axis=1)
                  / np.maximum(S - 1, 1), np.nan)
    inv_n = np.nansum(np.where(ok, 1.0 / np.maximum(nn, 1), 0), axis=1) / np.maximum(S, 1)
    if ref is None:
        dof = np.where(ok, nn - 1, 0)
        sig2 = np.nansum(np.where(ok, v * dof, 0), axis=1) / np.maximum(dof.sum(axis=1), 1)
        unit = mu
    else:
        sig2, unit = ref
    excess = np.sqrt(np.maximum(Vb - sig2 * inv_n, 0.0)) / np.abs(unit)

    w = np.where(ok, nn / np.where(np.isfinite(v), v, 1.0), 0.0)
    W = w.sum(axis=1)
    mt = np.where(W > 0, (w * np.nan_to_num(m)).sum(axis=1) / np.maximum(W, 1e-300), np.nan)
    num = (w * (np.nan_to_num(m) - mt[:, None]) ** 2).sum(axis=1) / np.maximum(S - 1, 1)
    lam = np.where(ok, (1.0 - w / np.maximum(W, 1e-300)[:, None]) ** 2
                   / np.maximum(nn - 1, 1), 0.0).sum(axis=1)
    F = num / (1.0 + 2.0 * (S - 2) / np.maximum(S ** 2 - 1, 1) * lam)

    bad = S < MIN_SECTORS
    return (np.where(bad, np.nan, excess), np.where(bad, np.nan, mu),
            np.where(bad, np.nan, F))


def gradient_leakage(ring_idx, sec_idx, q, keep, n_rings, n_sectors, mu):
    """Fake azimuthal scatter produced by the RADIAL gradient.

    Sectors do not share a mean q, so a ring with slope dI/dq shows sector-mean
    differences |dI/dq| * spread_s(q_s) even when perfectly homogeneous. This is
    the systematic floor of `excess`, and the quantity that should drive the
    choice of radial bin width.
    """
    n, sq, _ = cell_moments(ring_idx, sec_idx, q, keep, n_rings, n_sectors)
    ok = n >= N_MIN
    qbar = np.where(ok, sq / np.maximum(n, 1), np.nan)
    spread = np.nanstd(qbar, axis=1)
    qmid = np.nanmean(qbar, axis=1)
    with np.errstate(invalid="ignore"):
        grad = np.abs(np.gradient(mu, qmid))
    return grad * spread / np.abs(mu), qmid


# ==========================================================================
#  the frame every candidate is scored on
# ==========================================================================
@dataclass
class AzimuthalFrame:
    """Ring/sector geometry + intensities, fixed once per run.

    Built from the FLOOR-only pixel set so that rings, sectors, the
    winsorization thresholds and the per-ring `(sigma^2, mu)` reference are
    identical for every candidate -- otherwise a mask would be scored on a
    partition, and against a noise floor, it had itself chosen.
    """
    q: np.ndarray
    inten: np.ndarray
    clipped: np.ndarray
    usable: np.ndarray
    floor: np.ndarray
    ring_idx: np.ndarray
    sec_idx: np.ndarray
    n_rings: int
    n_sectors: int
    ref: tuple                      # frozen per-ring (sigma^2, mu)

    def excess(self, mask) -> np.ndarray:
        """Per-ring excess azimuthal scatter with `mask` removed."""
        keep = self.usable & ~np.asarray(mask, bool)
        e, _, _ = excess_scatter(*cell_moments(self.ring_idx, self.sec_idx,
                                               self.clipped, keep,
                                               self.n_rings, self.n_sectors),
                                 ref=self.ref)
        return e

    def random_control(self, mask, rng) -> np.ndarray:
        """A mask of the same size as `mask`, with its free pixels moved at random.

        Built as "the floor pixels this candidate actually masks, plus |mask \\
        floor| pixels drawn uniformly from outside the floor". Keeping the
        candidate's own floor part rather than the whole floor is what makes the
        control the same SIZE as the candidate even when the candidate does not
        contain the floor (an eroded mask does not) -- otherwise the control
        would mask more pixels than the candidate and win on volume alone.
        """
        mask = np.asarray(mask, bool)
        extra = int((mask & ~self.floor).sum())
        pool = np.flatnonzero((~self.floor & self.usable).ravel())
        out = (mask & self.floor).copy()
        if extra and pool.size:
            chosen = rng.choice(pool, size=min(extra, pool.size), replace=False)
            flat = out.ravel()
            flat[chosen] = True
            out = flat.reshape(mask.shape)
        return out


def build_frame(sample, floor, n_sectors: int = N_SECTORS) -> AzimuthalFrame:
    q, chi, inten, valid = pixel_frame(sample)
    usable = valid & np.isfinite(inten)
    n_rings = max(int(usable.sum() / (TARGET_CELL * n_sectors)), 8)
    edges = ring_edges(q, usable, n_rings)
    n_rings = len(edges) - 1
    ring_idx = np.clip(np.digitize(q, edges) - 1, 0, n_rings - 1)
    floor = np.asarray(floor, bool)
    sec_idx = sector_edges_per_ring(ring_idx, chi, usable & ~floor, n_rings, n_sectors)
    clipped = winsorize_per_ring(inten, ring_idx, usable, n_rings)
    ref = ring_reference(*cell_moments(ring_idx, sec_idx, clipped, usable & ~floor,
                                       n_rings, n_sectors))
    return AzimuthalFrame(q=q, inten=inten, clipped=clipped, usable=usable,
                          floor=floor, ring_idx=ring_idx,
                          sec_idx=sec_idx, n_rings=n_rings, n_sectors=n_sectors,
                          ref=ref)


def azimuthal_diagnostics(
    frame: AzimuthalFrame,
    mask: np.ndarray,
    rng: np.random.Generator,
    controls: int = 20,
) -> dict:
    """Candidate excess and repeated size-matched control comparisons.

    ``gain`` and ``win_rate`` contain one observation per independently drawn
    control, allowing the caller to report Monte Carlo uncertainty rather than
    treating one random mask as an exact reference.
    """
    if controls < 2:
        raise ValueError("azimuthal evaluation needs at least two controls")
    candidate = frame.excess(mask)
    finite_candidate = np.isfinite(candidate)
    if not finite_candidate.any():
        raise ValueError("the candidate has no scorable azimuthal ring")

    gain = np.empty(controls, dtype=np.float64)
    win_rate = np.empty(controls, dtype=np.float64)
    for i in range(controls):
        control = frame.excess(frame.random_control(mask, rng))
        valid = finite_candidate & np.isfinite(control)
        if not valid.any():
            raise ValueError("a matched control has no ring comparable to the candidate")
        gain[i] = float(np.median(control[valid] - candidate[valid]))
        win_rate[i] = float(np.mean(candidate[valid] < control[valid]))
    return {
        "excess": float(np.median(candidate[finite_candidate])),
        "gain": gain,
        "win_rate": win_rate,
    }
