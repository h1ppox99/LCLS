"""
identification/ -- the condition-contrast model of a bad pixel.

The label-free metrics in `automask.unsupervised` score a mask that already
exists. This package is about the step before that: writing down what a bad
pixel IS, as a statement about a forward model, and asking whether that
statement is identifiable from the data at all.

The model (see `docs/IDENTIFICATION.md` for the derivation):

    x_it = tau_i * F_t * A_i * S(q_i; theta_c(t))  +  F_t * J_i  +  eps_it

with `F_t` measured, `A_i` known from calibration, `S` the isotropic sample
signal, `tau_i <= 1` a static multiplicative shadow and `J_i >= 0` a static
additive parasitic term. The mask is `{tau != 1} | {J != 0}`.

Four modules, each carrying one part of the claim:

  `conditions` -- what plays the role of `c`, and the one XTC pass that reduces
                  a run to flux-normalised per-pixel means per condition group.
  `twoway`     -- the estimators: what is identifiable (only WITHIN-ring
                  structure), how much identifying power a run actually has,
                  and the two-stage tau -> J recovery.
  `harmonics`  -- the azimuthal harmonic cut that is supposed to separate a
                  sharp shadow from real sample anisotropy and from
                  polarization.
  `priors`     -- the three MAP detectors, one per artifact class, acting on a
                  continuous log-likelihood-ratio field instead of on a mask.

Nothing here is production. Every claim these modules encode is registered as a
falsifiable experiment in `automask.studies.loss_identification`, which is the
entry point.
"""
