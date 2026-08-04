"""
identification/forward.py -- the forward model, simulated, so the estimators can
be scored against a truth that exists.

Every claim in `twoway.py` is a claim about an estimator: that a contrast
cancels `J` exactly, that only within-ring structure is recoverable, that
precision is set by `Var_c(gamma)`. Those are checkable without any beamtime,
and checking them on a field where `tau`, `J`, `A` and `S` are known by
construction is strictly more informative than checking them on run 475, where
the answer is not known either. This module is that field.

It is deliberately faithful in the ways that matter to the estimators and
deliberately crude in the ways that do not:

  FAITHFUL   the geometry (beam near a corner, so rings are arcs and azimuthal
             coverage is partial -- the thing that caps the harmonic cut), the
             `2theta` range of the real detector, the polarization factor, the
             composition of the artifacts (`tau` multiplies the signal, `J` adds
             to it, and neither depends on the condition), and the noise, which
             is heteroscedastic in the mean the way photon counting is.
  CRUDE      the canvas is ~200 x 200 rather than 1064 x 1030, the sample signal
             is a smooth powder-like profile with no Bragg texture, and group
             means are drawn directly from their sampling distribution instead
             of being accumulated shot by shot.

The crudeness is bounded on purpose: no claim tested against this bench depends
on canvas size, and the two claims that DO depend on real sample texture (that
the residual anisotropy is low-order in `m`, and that real artifacts split into
multiplicative and additive classes) are marked in the study as requiring a run,
not simulated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

#: Real experiment geometry (CLAUDE.md / automask.azimuthal).
DIST_M = 0.190
WAVELENGTH_A = 1.2915
TWO_THETA_MAX = np.radians(28.9)      # q ~ 2.43 A^-1 at the far corner


@dataclass
class SynthConfig:
    shape: Tuple[int, int] = (220, 228)
    center: Tuple[float, float] = (214.0, 12.0)
    n_conditions: int = 4
    n_shots: int = 200
    n_rings: int = 24
    contrast: float = 0.25            # how much the conditions move S(q)
    level: float = 2.0e4              # signal scale, arbitrary flux-normalised units
    photon_scale: float = 1.0         # var = photon_scale * mean + read^2
    read: float = 12.0
    pol: float = 1.0
    shadow_transmission: float = 0.55
    shadow_center: Tuple[float, float] = (0.42, 0.55)
    shadow_radius: float = 14.0
    ring_shadow_transmission: float = 0.80
    ring_shadow_rings: Tuple[int, int] = (9, 11)
    streak_amplitude: float = 0.06    # as a fraction of the local signal
    streak_width: float = 2.2
    streak_angle_deg: float = 28.0
    gaps: bool = True
    noisy_sem: bool = True


@dataclass
class SynthTruth:
    q: np.ndarray
    chi: np.ndarray
    two_theta: np.ndarray
    A: np.ndarray
    solid_angle: np.ndarray
    tau: np.ndarray
    J: np.ndarray
    S: np.ndarray                     # (n_conditions, H, W) noiseless signal
    ring_idx: np.ndarray
    n_rings: int
    valid: np.ndarray
    shadow: np.ndarray
    ring_shadow: np.ndarray
    streak: np.ndarray
    cfg: SynthConfig = field(default_factory=SynthConfig)

    @property
    def bad(self) -> np.ndarray:
        """`{tau != 1} | {J != 0}` -- the mask the model defines."""
        return (self.shadow | self.ring_shadow | self.streak) & self.valid


def geometry(cfg: SynthConfig):
    """`(q, chi, two_theta)` on the synthetic canvas.

    The pixel size is chosen so the far corner lands at the real detector's
    maximum `2theta`, which is what makes the polarization depth and the
    azimuthal coverage of each ring representative.
    """
    rows, cols = np.indices(cfg.shape).astype(np.float64)
    dr = rows - cfg.center[0]
    dc = cols - cfg.center[1]
    r = np.hypot(dr, dc)
    chi = np.arctan2(dc, dr)
    two_theta = np.arctan(r / r.max() * np.tan(TWO_THETA_MAX))
    q = 4.0 * np.pi * np.sin(two_theta / 2.0) / WAVELENGTH_A
    return q, chi, two_theta


def known_correction(two_theta: np.ndarray, chi: np.ndarray, pol: float):
    """`A_i = Omega_i * P(chi_i, q_i)`, the part calibration is supposed to give.

    Returns `(A, solid_angle)` so an experiment can feed the estimator a
    DELIBERATELY incomplete `A` (solid angle only) and measure what an
    uncorrected polarization does to the recovered transmission.
    """
    omega = np.cos(two_theta) ** 3
    P = 1.0 - pol * np.sin(two_theta) ** 2 * np.cos(chi) ** 2
    return omega * P, omega


def equal_count_rings(q: np.ndarray, valid: np.ndarray, n_rings: int):
    v = np.sort(q[valid])
    edges = np.unique(v[np.linspace(0, v.size - 1, n_rings + 1).astype(int)])
    n = edges.size - 1
    return np.clip(np.digitize(q, edges) - 1, 0, n - 1), n


def _disk(shape, center_frac, radius):
    rows, cols = np.indices(shape).astype(np.float64)
    return ((rows - center_frac[0] * shape[0]) ** 2
            + (cols - center_frac[1] * shape[1]) ** 2) <= radius ** 2


def _band(shape, angle_deg, width, offset_frac=0.35):
    rows, cols = np.indices(shape).astype(np.float64)
    th = np.radians(angle_deg)
    d = ((cols - offset_frac * shape[1]) * np.cos(th)
         - (rows - 0.5 * shape[0]) * np.sin(th))
    return np.abs(d) <= width / 2.0


def signal(q: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """`S(q; theta_c)` for every condition, `(G, H, W)`.

    A falling powder-like profile times a condition-dependent, q-dependent
    modulation. `cfg.contrast = 0` makes every condition identical, which is the
    degenerate case the identifying-power measurement has to detect.
    """
    base = cfg.level * (q + 0.25) ** -2.2
    out = []
    for c in range(cfg.n_conditions):
        phase = 2.0 * np.pi * c / max(cfg.n_conditions, 1)
        out.append(base * (1.0 + cfg.contrast * np.cos(2.2 * q + phase)))
    return np.stack(out, axis=0)


def truth(cfg: Optional[SynthConfig] = None) -> SynthTruth:
    """Everything the simulator knows, with no noise applied."""
    cfg = cfg or SynthConfig()
    q, chi, two_theta = geometry(cfg)
    A, omega = known_correction(two_theta, chi, cfg.pol)

    valid = np.ones(cfg.shape, bool)
    if cfg.gaps:
        valid[:, 108:114] = False          # a panel seam
        valid[103:109, :] = False
    ring_idx, n_rings = equal_count_rings(q, valid, cfg.n_rings)

    shadow = _disk(cfg.shape, cfg.shadow_center, cfg.shadow_radius) & valid
    ring_shadow = (np.isin(ring_idx, np.arange(*cfg.ring_shadow_rings))
                   & valid & ~shadow)
    streak = _band(cfg.shape, cfg.streak_angle_deg, cfg.streak_width) & valid
    streak &= ~shadow & ~ring_shadow

    tau = np.ones(cfg.shape)
    tau[shadow] = cfg.shadow_transmission
    tau[ring_shadow] = cfg.ring_shadow_transmission

    S = signal(q, cfg)
    J = np.zeros(cfg.shape)
    J[streak] = cfg.streak_amplitude * (tau * A * S[0])[streak]

    return SynthTruth(q=q, chi=chi, two_theta=two_theta, A=A, solid_angle=omega,
                      tau=tau, J=J, S=S, ring_idx=ring_idx, n_rings=n_rings,
                      valid=valid, shadow=shadow, ring_shadow=ring_shadow,
                      streak=streak, cfg=cfg)


def simulate(cfg: Optional[SynthConfig] = None,
             rng: Optional[np.random.Generator] = None,
             t: Optional[SynthTruth] = None):
    """Draw condition means and their standard errors from the forward model.

    Returns `(zbar, sem2, truth)` with `zbar` and `sem2` shaped `(G, H, W)` --
    the same pair `identification.conditions.GroupMoments` serves from a real
    run, so every estimator takes the two interchangeably.

    `sem2` is the ESTIMATED sampling variance, not the exact one: with
    `noisy_sem` on it carries the chi-square scatter a variance estimated from
    `n_shots` frames actually has. An estimator whose behaviour depends on
    having an exact noise model would pass with the exact version and fail on a
    run, so the bench does not hand it one.
    """
    t = t or truth(cfg)
    cfg = t.cfg
    rng = rng or np.random.default_rng(0)

    clean = t.tau[None] * t.A[None] * t.S + t.J[None]
    var_shot = cfg.photon_scale * np.maximum(clean, 0.0) + cfg.read ** 2
    sem2_true = var_shot / cfg.n_shots
    zbar = clean + rng.normal(0.0, np.sqrt(sem2_true))

    dof = max(cfg.n_shots - 1, 1)
    sem2 = (sem2_true * rng.chisquare(dof, size=sem2_true.shape) / dof
            if cfg.noisy_sem else sem2_true)
    bad = ~np.broadcast_to(t.valid, zbar.shape)
    return np.where(bad, np.nan, zbar), np.where(bad, np.nan, sem2), t
