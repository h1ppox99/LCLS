---
name: xray-masking-azimuthal-residual
description: Signal-based mask for DIFFUSE azimuthal-symmetry-breaking features (parasitic-scattering blobs, ghosts, window scatter) on the assembled sum — radial-median model, destriped residual, matched-scale smoothing, hysteresis growth. Catches low-contrast extended anomalies that per-pixel sigma-clip (method 02) cannot see; validated on the Run0475 off-ring blob at (row 686, col 387).
category: masking
role: signal-dependent
gate: diffuse parasitic blobs / scatter ghosts on a summed assembled image where powder symmetry holds
status: wired
---

# Masking · 05 — azimuthal-residual blob detection (signal-based)

**Complements** [02_pyfai_azimuthal_sigmaclip.md](02_pyfai_azimuthal_sigmaclip.md): method 02
clips *per-pixel* outliers versus the ring statistics, so it needs the outlier to stand out of
the single-pixel noise. A diffuse blob (e.g. FWHM ~33 px, +70% over background but only
~0.7 σ_pixel per pixel) is invisible to it. This method detects the *envelope* instead.

## Principle

For powder/isotropic scattering the expected image depends on radius only. Build the
azimuthal-median model `I_model(r)`, subtract, remove the detector's separable row/column
stripe structure, then smooth at the scale of the feature you hunt (matched filter) and
threshold a robust z-score. Grow the detections with hysteresis so the full diffuse envelope
is masked, not just the core.

```
model    : I_model(r) = median over azimuth in 1-px radial bins (3-bin boxcar)
residual : R = img − I_model(r)
destripe : R ← R − median_per_column(R);  R ← R − median_per_row(R)      # CRITICAL
smooth   : S = gaussian(R, σ_smooth) / gaussian(coverage, σ_smooth)      # NaN-aware
z-score  : z = (S − median(S)) / (1.4826 · MAD(S))
detect   : seeds = components(z > z_seed) with area ≥ min_size
grow     : layer = binary_propagation(seeds, support = z > z_grow)
policy   : keep components with |r_comp − r_ring| > ring_margin (off-ring only), or keep all
refine   : per kept component, fit a 2-D Gaussian to the destriped residual and extend the
           mask to the ellipse where the fitted excess > ε · background  (contamination contour)
```

The grow step alone recovers only the smoothed envelope (~the FWHM core). A diffuse
feature's wings still bias the azimuthal mean — on Run0475 the envelope-only mask left a
**1.5 % per-bin residual bias**, larger than the background noise floor (1.2 %). The
parametric refinement masks out to a *stated contamination level* instead of a z-contour.

Masked pixels are the grown components; union onto the run's other mask layers.

## Parameters

Tested on Run0475, agent_trial_02 sum:

| Param | Value | Why |
|---|---|---|
| radial bins | 1 px, azimuthal **median**, 3-bin boxcar | a blob spanning ~1–2% of its ring's azimuth cannot bias the median model |
| destriping | per-column then per-row residual medians | **decisive**: residual noise 69 → 31 ADU; without it the Run0475 blob peaks at z=3.8 (missed at any sane threshold), with it z=6.2 |
| `σ_smooth` | 8 px | matched to FWHM ~33 px targets (σ ≈ FWHM/4); sensitive to ~15–100 px features |
| `z_seed` | 5.0 | no false seeds on 1.1 Mpx after smoothing (Gaussian tail ≪ 1 px expected) |
| `min_size` | 150 px | single-pixel/small residues are method 02 / status territory |
| `z_grow` | 3.0 | hysteresis growth covers the diffuse envelope (905 px for the Run0475 blob) |
| ring policy | `off-ring` (margin 60 px) for parasitic cleanup | on-ring detections are Bragg/texture — a separate decision (see trade-offs) |
| `r_min` | 100 px | components nearer the beam center are beamstop/direct-beam territory (layers 00/geometry/beamstop), not blobs — without this, Run0475 grows a spurious 779-px component at r ≈ 35 |
| `refine_contour_eps` | 0.10 | mask the fitted-Gaussian ellipse out to excess = ε · background. **Rule: residual per-bin bias must stay below HALF the background noise floor; take the largest ε that satisfies it** — masking further buys nothing measurable. Run0475 sweep (bias vs the 1.2 % floor): envelope-only 1.5–1.9 % (ABOVE floor — invalid), ε=0.15 → 0.88 % (floor/1.4 — too close, invalid), **ε=0.10 → 0.52 % (floor/2.3, default, 2 066 px)**, ε=0.05 → 0.26 % (floor/4.7, 2 767 px), ε=0.02 → 0.11 % (3 715 px, precision work). Set `None` to disable |
| image center | load `image_center.json`, assembled `(row, col)` | shared run artifact; never a method default |
| disjointness | `layer &= ~base_mask` | smoothing interpolates z across masked speckles, so grown components can leak a few px into already-masked area; subtracting keeps per-layer accounting exact |

## Implementation

Reference implementation (numpy/scipy only):

```python
import numpy as np
from scipy import ndimage

def azimuthal_residual_layer(img, base_mask, bc,
                             sigma_smooth=8.0, z_seed=5.0, z_grow=3.0,
                             min_size=150, ring_radii=(367, 531, 734, 847, 1245),
                             ring_margin=60, off_ring_only=True, r_min=100.0,
                             refine_contour_eps=0.10):
    safe = np.where(base_mask | (np.abs(img) > 1e5), np.nan, img)
    H, W = img.shape
    rr, cc = np.mgrid[0:H, 0:W]
    rad = np.hypot(rr - bc[0], cc - bc[1]); rbin = rad.astype(int)
    ok = ~np.isnan(safe)
    med_r = np.full(rbin.max() + 1, np.nan)
    for i in np.unique(rbin[ok]):
        sel = ok & (rbin == i)
        if sel.sum() >= 40: med_r[i] = np.median(safe[sel])
    v = ~np.isnan(med_r); k = np.ones(3) / 3
    med_s = med_r.copy()
    med_s[v] = np.convolve(np.nan_to_num(med_r), k, "same")[v] / \
               np.maximum(np.convolve(v.astype(float), k, "same")[v], 1e-9)
    resid = safe - med_s[rbin]
    resid -= np.nanmedian(resid, axis=0)[None, :]          # destripe cols
    resid -= np.nanmedian(resid, axis=1)[:, None]          # destripe rows
    ok = ~np.isnan(resid)
    sm = ndimage.gaussian_filter(np.nan_to_num(resid), sigma_smooth)
    w = ndimage.gaussian_filter(ok.astype(float), sigma_smooth)
    sm = np.where(w > 0.6, sm / np.maximum(w, 1e-9), np.nan)
    zmap = (sm - np.nanmedian(sm)) / (1.4826 * np.nanmedian(np.abs(sm - np.nanmedian(sm))))
    zf = np.nan_to_num(zmap)
    lab, n = ndimage.label(zf > z_seed)
    sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
    seeds = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_size])
    grown = ndimage.binary_propagation(seeds, mask=zf > z_grow)
    layer = np.zeros_like(base_mask)
    glab, gn = ndimage.label(grown)
    kept = []
    for g in range(1, gn + 1):
        comp = glab == g
        r_comp = rad[comp].mean()
        if r_comp < r_min:
            continue                       # beam-center territory: other layers' job
        on_ring = any(abs(r_comp - rp) <= ring_margin for rp in ring_radii)
        if (not off_ring_only) or (not on_ring):
            kept.append(comp)
            layer |= comp
    if refine_contour_eps:                 # parametric contamination-contour refinement
        from scipy.optimize import least_squares
        for comp in kept:
            ys, xs = np.where(comp)
            y0g, x0g = float(ys.mean()), float(xs.mean())
            r_eq = np.sqrt(comp.sum() / np.pi)          # component-equivalent radius
            half = int(np.clip(2.5 * r_eq + 2 * sigma_smooth, 60, 300))
            sl = (slice(max(0, int(y0g) - half), min(H, int(y0g) + half)),
                  slice(max(0, int(x0g) - half), min(W, int(x0g) + half)))
            # Fit on the SMOOTHED field, not the raw residual: it is nearly noise-free
            # and continuous, so the fit cannot fragment on mask gaps or lock into
            # narrow robust-loss minima. The known kernel is then deconvolved exactly.
            Zc, Yc, Xc = sm[sl], rr[sl], cc[sl]
            good = np.isfinite(Zc)
            if good.sum() < 500:
                continue
            def gmod(p, x, y):
                A, x0, y0, sa, sb, off = p
                return A * np.exp(-(x - x0)**2 / (2 * sa**2)
                                  - (y - y0)**2 / (2 * sb**2)) + off
            s0 = float(np.clip(r_eq, 1.2 * sigma_smooth, 100.0))
            try:
                p = least_squares(
                    lambda p: gmod(p, Xc[good], Yc[good]) - Zc[good],
                    [float(np.nanmax(Zc)), x0g, y0g, s0, s0, 0.0],
                    loss="soft_l1", f_scale=30).x
            except Exception:
                continue
            A_f, x0, y0, sa_f, sb_f, _ = p
            # deconvolve the smoothing kernel: sigma_true^2 = sigma_fit^2 - sigma_smooth^2
            sa_t = np.sqrt(max(sa_f**2 - sigma_smooth**2, 9.0))
            sb_t = np.sqrt(max(sb_f**2 - sigma_smooth**2, 9.0))
            A_t = A_f * (abs(sa_f) * abs(sb_f)) / (sa_t * sb_t)
            bg = med_s[min(int(np.hypot(y0 - bc[0], x0 - bc[1])), len(med_s) - 1)]
            if not (np.isfinite(bg) and bg > 0 and A_t > refine_contour_eps * bg):
                continue
            g_r = np.sqrt(2 * np.log(A_t / (refine_contour_eps * bg)))
            layer |= (((cc - x0) / (sa_t * g_r))**2
                      + ((rr - y0) / (sb_t * g_r))**2) <= 1
    layer &= ~base_mask                    # keep layers disjoint for exact accounting
    return layer, zmap
```

## Evidence (Run0475)

### Scale range (validated by synthetic injection at r=648, off-ring)

Gaussian blobs injected into the real Run0475 sum, default `sigma_smooth=8`:

| injected σ_b (FWHM) | contrast | detected? | footprint vs true-2% region |
|---|---|---|---|
| 13 (31 px) — the real blob | 77 % of bg | z = 6.2 ✓ | ~100 % (reference) |
| 30 (71 px) | 77 % | z = 6.7 ✓ | 80 % (474 px overreach) |
| 60 (141 px) | 77 % | z = 10.5 ✓ — bigger = easier | 96.7 % |
| 60 (141 px) | **15 %** | ✗ at any σ_smooth | — |
| 120 (283 px) | 40 % | ✗ | — |

Three boundaries this establishes:

- **Bigger same-contrast blobs are EASIER** (more pixels aggregate) — no σ_smooth retune
  needed up to FWHM ≈ 140 px. Enlarging σ_smooth is counterproductive (σ=30 drops the
  σ_b=60 blob from z=10.5 to 5.6): the ~31 ADU floor is *structured* residue, which does
  not average down like white noise.
- **Amplitude wall:** smoothed-peak must exceed ~5× the 31 ADU floor → minimum detectable
  contrast ≈ 45 % of background at these scales. A 15 %-contrast feature is invisible to
  any smoothing choice — that regime needs cross-run differencing.
- **Size ceiling:** at σ_b ≈ 120 the feature occupies ~26 % of its ring's covered arc — the
  azimuthal median starts absorbing it (self-subtraction) and the grown component overlaps
  ring bands, so the off-ring policy rejects it. Beyond this, iterative model
  re-estimation (detect → mask → refit model) or a reference run is required.

Refinement note: the fit runs on the **smoothed field** with exact kernel deconvolution
(σ_true² = σ_fit² − σ_smooth²) — fitting the raw residual fragments on mask gaps for large
components; a nonparametric contour is noise-limited below ε ≈ σ_floor/bg ≈ 9 %.

### Result (agent_trial_02 sum, off-ring policy)

Exactly **one** off-ring component: the parasitic blob at (row 686, col 387), radius 466 px,
azimuth −42° — peak z = 6.2. Envelope (grow) stage: ~900 px covering the FWHM core, which
still leaves a **1.54 % per-bin bias** in I(q) from the unmasked wings. Parametric
refinement (2-D Gaussian fit on the smoothed field with kernel deconvolution:
A ≈ 309 ADU = 77 % of background, σ ≈ 13×12 px; ellipse at the default 10 % contamination
contour) sizes the layer at **≈2 100 px (0.20 % of panel area, r ≈ 26 px)** with residual
per-bin bias **≈0.52 %** — comfortably below the floor/2 ceiling. (ε=5 % → ≈2 800 px /
0.26 %; ε=2 % → ≈3 700 px / 0.11 % for precision line-shape work.)
On-ring detections (12 Bragg spots at r ≈ 846–849 and outer-arc segments at r ≈ 1243–1245)
are reported but not masked under `off_ring_only=True`; the near-beam-center residual at
r ≈ 35 is excluded by `r_min`.
Independent evidence: 2° azimuthal-bin medians in the radial band 441–491 put the blob's
sector at **+4.7 σ** above the band median.

## When to use

Cleaning diffuse, spatially-fixed parasitic features from a summed/averaged assembled image
where powder symmetry holds: scatter ghosts from windows/kapton, beamline reflections,
extended stray-light patches. Run it on the *sum*, after selection + normalization.

## Trade-offs

Signal-dependent — with `off_ring_only=False` it will also mask real anisotropic signal
(Bragg spots from large grains, texture arcs); keep the off-ring policy unless the goal is a
strictly isotropic background. Needs an image center (~few-px accuracy is enough at these
radii). `σ_smooth` sets the sensitive size band — rescan it for targets much smaller/larger
than ~30 px. Destriping assumes stripes are separable row/column structure (true for
Jungfrau column noise); pathological large-area gradients would need a 2-D background model
instead.

## Outputs

Blob-mask layer (grown components, ε-contour refined), unioned onto the run's other
mask layers. Provenance: validated in `outputs/agent_trial_02/`: `_blob_proposal.png`
(evidence figure), `_blob_det_z2.npy` (z-map). First applied run:
`outputs/agent_trial_02_azres/`.

## Links

Part of: [masking](README.md). Complements [02](02_pyfai_azimuthal_sigmaclip.md)
(per-pixel clip). Screener front-end: [06](06_azimuthal_sector_itheta.md) finds the
violations; this method builds the footprint for blob-like positives.
