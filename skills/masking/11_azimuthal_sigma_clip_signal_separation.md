---
name: xray-masking-azimuthal-sigma-clip-signal-separation
description: Literature method (Kieffer et al. 2025, J. Appl. Cryst. 58, 138; arXiv:2411.09515) — iterative sigma-clipping in AZIMUTHAL space separates the amorphous/powder background from single-crystal Bragg signal at 900+ Hz. Per q-ring, reject |I - <I>| > n*sigma and re-fit until no new outliers; SIGNED outliers carry meaning — positive = Bragg peaks, negative = shadows or defective pixels. Replaces the older azimuthal median filter (which is sort-bound and produces bin-to-bin jumps). Paper — papers/masking/08_Kieffer_2025_Signal_Separation_Sigma_Clipping.pdf.
category: masking
role: signal-dependent (literature method — the source of method 02)
gate: azimuthal symmetry holds (powder/amorphous background); needs a calibrated q-map
status: wired (as method 02)
---

# Masking · 11 — Azimuthal sigma-clipping / signal separation (Kieffer et al. 2025)

Literature source for this repo's [02_pyfai_azimuthal_sigmaclip](02_pyfai_azimuthal_sigmaclip.md),
and the current state of the art for what pyFAI's clipping actually does. Written for
real-time use on a Jungfrau 4M at ~1 kHz, so its parameter advice is about convergence and
cost, not just correctness.

## Principle

Work in **azimuthal space**: pixels on the same q-ring should be samples of one
distribution. Its histogram is a bell with a few positive outliers (the Bragg peaks). So:

```
per q-ring, iterate:
    reject pixels with  |I − ⟨I⟩| > n·σ(I)
    recompute ⟨I⟩, σ  on the survivors
until no new outliers are found
```

Two points the paper makes that change how the method should be used:

1. **The mean/σ are outlier-sensitive by construction** — few Bragg pixels inflate σ a lot.
   That is why clipping must **iterate**: each pass re-centres the distribution of the
   remaining pixels. A single pass under-rejects.
2. **The sign of the outlier is information.** Positive outliers → Bragg peaks; **negative
   outliers → shadows or defective pixels**. A masking workflow that only looks at positive
   deviations throws away the half that points at detector problems.

It supersedes the earlier **azimuthal median filter** (pyFAI 2013) for two concrete reasons
given in the paper: the median is *computationally heavy* (every azimuthal bin must be
sorted), and the resulting 1D curve shows **sharp jumps from one azimuthal bin to its
neighbour**.

## Parameters

| Param | Guidance |
|---|---|
| cut-off `n` (SNR) | 3.0 typical; smaller = more aggressive |
| iterations | **iterate until no new outliers** — the paper's rule; a fixed small count under-rejects, and runtime scales with the count |
| error model | use the in-ring (`azimuthal`) variance when data are averaged ADU, not photon counts |
| pixel splitting | off — splitting puts one pixel in several rings and breaks the clip logic |
| q-map | must be calibrated; the method is only as good as the geometry |

## Decision rules

- Use when azimuthal symmetry genuinely holds. On textured or single-crystal samples the
  "outliers" are the science.
- **Keep both signs.** Route positive detections as Bragg/texture (do not mask them unless
  the goal is a strictly isotropic background) and negative detections to the defect/shadow
  discussion — this is the same two-signed discipline as [06](06_azimuthal_sector_itheta.md).
- Known shadows are better handled by an explicit user mask than by clipping; the paper is
  candid that background *anisotropy* (e.g. stretched plastic films) is **not** addressable
  this way at all.

## Evidence (Run0475)

This repo's implementation ([02](02_pyfai_azimuthal_sigmaclip.md)) flags 9 696 / 1 048 576 px
(**0.925 %**) at `thres` 3.0, `max_iter` 5, `error_model="azimuthal"`, no pixel splitting.
Cross-checked against the signal-independent RMM mask: ~8 035 of those px are real
Bragg/texture, i.e. the price of a signal-dependent mask — exactly the trade-off this paper
frames as *separation* rather than masking. The paper's negative-outlier insight is what
[06](06_azimuthal_sector_itheta.md) generalizes here to two-signed sector screening (which
found the ASIC-block deficits at r ≈ 284 and 1001 px that positive-only detection misses).

## When to use

Cleaning an isotropic/amorphous or powder background; separating amorphous background from
Bragg signal for compression or hit-finding; any place a per-ring robust statistic is wanted
at frame rate.

## Trade-offs

Removes real anisotropic signal if applied unguarded. Requires correct geometry. Iterating to
convergence costs time — the very reason the paper engineers it carefully for kHz rates.
Background anisotropy is out of scope by the authors' own statement.

## Outputs

Per-ring clipped background + an outlier map separable by sign. Paper:
`papers/masking/08_Kieffer_2025_Signal_Separation_Sigma_Clipping.pdf`
(doi:10.1107/S1600576724011038; arXiv:2411.09515, open access).

## Links

Part of: [masking](README.md). Implemented here as [02](02_pyfai_azimuthal_sigmaclip.md).
Two-signed generalization: [06](06_azimuthal_sector_itheta.md); diffuse-envelope complement:
[05](05_azimuthal_residual.md).
