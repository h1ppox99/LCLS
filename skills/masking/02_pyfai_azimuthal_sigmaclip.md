---
name: xray-masking-pyfai-azimuthal-sigmaclip
description: Signal-based mask via pyFAI azimuthal (χ) sigma-clipping — flags pixels that deviate from the robust per-q-ring model (bad columns, cosmics, Bragg spots from large grains). Use to clean an isotropic/amorphous/powder background where azimuthal symmetry holds. Warning — it removes real anisotropic (single-crystal/texture) signal.
category: masking
role: signal-dependent
gate: isotropic/amorphous or powder background where azimuthal symmetry holds
status: wired
---

# Masking · 02 — pyFAI azimuthal sigma-clipping (signal-based)

## Principle

Diffraction has azimuthal (χ) symmetry: pixels on the same `q` ring should read similar
intensity. Iterate per ring — compute robust mean μ_q and std σ_q, clip pixels with
`|I − μ_q| > thres·σ_q`, recompute, repeat. Then back-project the 1D clean model to 2D via
`calcfrom1d` and flag any pixel with `dev = |I − mean_2D| / std_2D > thres`.

```
for each q-ring:  μ_q, σ_q = robust mean/std;  drop |I-μ_q| > thres·σ_q;  iterate ≤ max_iter
mask = ( |I_pixel - mean_2D(q)| / std_2D(q) ) > thres
```

## Parameters

Tested values, and why:

| Param | Value | Why |
|---|---|---|
| `thres` | 3.0 | Standard 3σ clip — balance removing outliers vs keeping signal. Smaller ⇒ more aggressive. |
| `max_iter` | 5 | Usually converged by 5; more over-clips real peaks. |
| `error_model` | `azimuthal` | Use in-ring variance, **not** Poisson — data are averaged ADU, not photon counts. |
| `npt_radial` | 122 | Matches the file's native azav q-grid (Δq ≈ 0.02 Å⁻¹) for direct comparison. |
| `radial_range` | [0.0, 2.44] Å⁻¹ | Covers the detector's full valid q. |
| `method` | `("no","csr","cython")` | `no` pixel-splitting is **required** — splitting puts one pixel in several rings and breaks the clip logic. |
| `correctSolidAngle` | True | Standard flat-detector correction (geometric corrections live in these pyFAI settings — see [normalization](../normalization/README.md)). |
| `polarization_factor` | None | Off in this demo to avoid assuming detector orientation; ~0.99 for quantitative work. |

**Geometry** (needed to build the q-map): dist 190 mm, λ 1.2915 Å (9.6 keV), beam center
(35.51, −35.22) mm, 75 µm pixels, `rot1=rot2=rot3=0`. Verified against stored `matrix_q`
to 4×10⁻¹² Å⁻¹.

## Evidence (Run0475)

9 696 / 1 048 576 ≈ **0.925 %** masked — bad columns, hotspots, and Bragg spots from large
grains sitting on the rings.

## When to use

Cleaning an isotropic/amorphous or powder background where azimuthal symmetry holds; catches
dynamic outliers (cosmics, sporadic Bragg) that dark masks cannot.

## Trade-offs

**Will remove real anisotropic signal** (single-crystal Bragg, texture). Requires correct
geometry (dist, λ, beam center) to build the q-map.

## Outputs

Sigma-clip mask layer. Provenance: `masking_pyFAI_sigmaclip/` — `build_mask.py`,
`mask.npy`, `sigmaclip_mask.npz`, and `README_掩膜方法说明.md` with the full geometry
and figure set.

## Links

Part of: [masking](README.md). Geometric corrections live in the pyFAI settings —
see [normalization](../normalization/README.md). Signal-independent complement:
[03](03_rmm_dark_based.md).
