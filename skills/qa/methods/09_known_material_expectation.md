---
name: xray-qa-known-material-expectation
description: Endpoint I(q) check against first-principles expectations of a KNOWN material (已知材料属性下的 quality checker) — the registry carries EXPLICIT theoretical q values for every reflection of every material; a material-agnostic D-voting ratio algorithm assigns observed sharp peaks to reflections (distance-free identity), derives the effective sample-detector distance as a geometry witness, enforces completeness over the FULL valid q-range, and flags extra sharp features as parasitics. Registry — LaB6 (validated on Run0475: (100)@847px, (110)@1245px, D_eff 196.9 mm), Si, CeO2, AgBh, Al2O3 — all judged by the same generic machinery.
category: qa
role: material-expectation (endpoint-level)
gate: sample identity known AND the material is present in this file's registry
status: not-wired
---

# QA · 09 — Known-material expectation (已知材料属性下的 quality checker)

Generalizes [06_calib_lab6_drift](06_calib_lab6_drift.md) from "are the found rings
where the lattice predicts" to "is the WHOLE curve what this material must produce":
completeness (every reachable reflection accounted for), identity (distance-free ratio
test), strength, width class, and nothing sharp that the material cannot explain.
Consumes [02_attr_feature_width](02_attr_feature_width.md) labels.

## Principle

When the sample is a known material, I(q) is predictable from first principles, and QA
can judge the endpoint against physics instead of only against thresholds calibrated on
past runs. Three layers of prediction, strongest first:

1. **Lattice rings.** For a crystalline calibrant every allowed reflection has an exact
   position: cubic `q_hkl = (2π/a)·√N`, `N = h²+k²+l²` (allowed N per centering);
   hexagonal via `1/d² = 4/3·(h²+hk+k²)/a² + l²/c²`; lamellar (AgBh) `q_n = n·q₁`.
   Every allowed reflection whose predicted position falls inside the **full valid
   q-range of the curve** (golden rule 5 — the range comes from the data, never from a
   hard-coded cap) must be found, or its absence explained (weak structure factor,
   masked arc). *A missing reachable ring is a finding, not a formatting choice* — the
   1245 px ring was invisible for months because a plot cap excluded it.
2. **The distance-free identity test — material-agnostic by construction.** On a flat
   detector, the ratio of two ring radii `r_i/r_j = tan(2θ_i)/tan(2θ_j)` depends only
   on λ and the material's q values — not on distance, not on the beam-center
   magnitude. The generic assignment is **D-voting**: every (observed sharp peak,
   registry reflection) pairing implies a candidate distance
   `D_cand = r·pix/tan(2θ_pred)`; the D value that ≥ 2 peaks agree on (within
   `ratio_tol_pct`) wins, and that agreement *is* the identity confirmation. The same
   machinery runs unchanged on LaB6, Si, CeO2, AgBh, or Al2O3 — only the registry's q
   table changes. The winning `D_eff` then becomes a **geometry witness**: its spread
   across rings measures geometry-model error, and D_eff vs the pipeline's nominal
   distance audits the q-axis calibration itself.
3. **Empirical anchors.** Features validated on this beamline that are *not* lattice
   rings (diffuse instrument/sample-environment scattering) are still stable witnesses
   of a healthy sum — checked by position and contrast, but classified `diffuse_anchor`,
   never "Bragg ring". Width is the discriminator, as in attribution: lattice rings must
   be sharp; anchors labeled diffuse must be broad.

Anything **sharp and predicted by neither layer** is a parasitic candidate (ice, window
crystallites) — routed exactly as attribution's `unattributed_sharp_feature`.

## Parameters

| Field | Default | Meaning |
|---|---|---|
| `material` | — (required) | registry key, passed by the orchestrator (e.g. `LaB6`) |
| `anchor_delta_px_max` | 5 | position tolerance for validated anchors/rings (px) |
| `anchor_geometry_tol_pct` | 2.0 | validated px anchors are geometry-bound: they gate only when this run's D_eff matches the registry's `validated_deff_mm` within this tolerance (otherwise skipped loudly) |
| `d_vote_bounds_x_nominal` | [0.5, 2.0] | candidate-D plausibility window as multiples of the nominal distance — junk peaks must not elect an absurd geometry; within the window the vote is distance-free (ties broken by smaller D spread) |
| `ratio_tol_pct` | 0.5 | tan-ratio identity-test tolerance (%) |
| `deff_spread_pct_max` | 0.5 | max spread of per-ring D_eff (geometry-model witness) |
| `deff_vs_nominal_warn_pct` | 1.0 | D_eff vs nominal distance beyond this → geometry-constant escalation |
| `sharp_fwhm_max_px` | 15 | width boundary: lattice rings must be at or below, diffuse anchors above |
| `extra_feature_prominence_sigma` | 6 | detection floor for the extra-sharp-feature scan (robust σ from MAD(ΔI)) |
| `completeness_edge_margin_pct` | 0.5 | reflections predicted within this margin of q_max are edge cases, not hard absences |

The search range is the curve's full valid extent (last bin with `n_valid ≥` the
integrator's floor) — never a fixed number.

### Registry: theoretical q values (Å⁻¹) — the source of truth

Explicit per-reflection theory, computed once from the lattice constants below and
frozen here (script consumes the machine block verbatim; `a`/`c` are provenance):

| Material | Lattice | Reflections (hkl: q / Å⁻¹) |
|---|---|---|
| **LaB6** (a = 4.15695 Å, cubic P) | all N except 7, 15, … | 100: 1.5115 · 110: 2.1376 · 111: 2.6180 · 200: 3.0230 · 210: 3.3798 · 211: 3.7024 · 220: 4.2751 · 300: 4.5345 · 310: 4.7797 |
| **Si** (a = 5.43102 Å, diamond cubic) | all-odd, or all-even with h+k+l=4n | 111: 2.0038 · 220: 3.2722 · 311: 3.8370 · 400: 4.6276 · 331: 5.0428 |
| **CeO2** (a = 5.41165 Å, fluorite F) | h,k,l same parity | 111: 2.0110 · 200: 2.3221 · 220: 3.2839 · 311: 3.8508 · 222: 4.0220 · 400: 4.6442 |
| **AgBh** (d₀₀₁ = 58.380 Å, lamellar) | equally spaced orders | 001: 0.1076 · 002: 0.2153 · 003: 0.3229 · 004: 0.4305 · 005: 0.5381 · 006: 0.6458 · 007: 0.7534 · 008: 0.8610 · 009: 0.9686 · 0010: 1.0763 |
| **Al2O3** corundum (a = 4.7591, c = 12.9918 Å, hexagonal) | strong lines | 012: 1.8054 · 104: 2.4630 · 110: 2.6405 · 006: 2.9018 · 113: 3.0128 · 024: 3.6109 · 116: 3.9233 · 300: 4.5735 |

## Decision rules

1. Resolve `material` in the registry — unknown → **skip loudly**
   (`material_not_in_registry`), never guess.
2. Determine the full valid r/q range from the curve itself.
3. **Identity (hard)**: assign observed sharp peaks to the registry's q values by
   **D-voting** — each (peak, reflection) pairing implies a candidate distance; the
   distance ≥ 2 peaks agree on within `ratio_tol_pct` fixes the assignment and
   confirms identity, with no geometry input at all. No consistent assignment →
   `material_identity_mismatch` — wrong sample in the beam, wrong λ, or a wrong
   registry entry; stop-the-line. **Degenerate case**: if only one sharp peak is
   reachable (e.g. Si below q ≈ 3.3 has only (111)), identity is underdetermined —
   record `identity_underdetermined` (soft), fall back to a position-only match at
   the nominal distance, and say so.
4. **Completeness (hard)**: an allowed reflection predicted inside the valid range but
   absent → `expected_lattice_ring_absent`. Before escalating, check whether the mask
   ate its arc (`panel_mask_fraction`, mask layers near that radius) — if so, route to
   **mask**; otherwise **reduction**.
5. **Anchors (per-anchor)**: each validated anchor found within `anchor_delta_px_max`
   with contrast ≥ its floor — same contract as verification C1/C2.
6. **Geometry witness (soft)**: D_eff spread > `deff_spread_pct_max` → geometry-model
   error (tilt, beam center) — hand the per-ring table to
   [06_calib_lab6_drift](06_calib_lab6_drift.md), which owns the hard q-scale verdict.
   |D_eff − D_nominal| > `deff_vs_nominal_warn_pct` → `geometry_constant_mismatch`:
   the q-axis labels are systematically off even though px-space anchors pass — route
   to calibration, do **not** fail the endpoint on it.
7. **Extras (soft)**: sharp features predicted by neither layer →
   `unattributed_sharp_feature`, with the same operational reading as attribution
   (on amorphous campaigns: ice/crystallite watch).
8. **Width consistency (soft)**: a "lattice ring" that is broad, or a "diffuse anchor"
   that is needle-sharp, means the registry label or the data is wrong — report, don't
   silently relabel.

## Implementation

`scripts/material_expectation.py` — reads the registry and thresholds from THIS file's
json block (same pattern as `entrypoint.py` → verification thresholds), runs all checks
on an `iq.npy`:

```bash
python3 skills/qa/scripts/material_expectation.py \
    --out-dir outputs/agent_trial_07 --material LaB6 --nominal-dist-mm 190
```

## Evidence (Run0475)

Worked indexing of the five historical "LaB6 rings" (anchors at 367 / 531 / 734 / 847 /
1245 px), λ = 1.2915 Å, a = 4.15695 Å:

- **Only two are lattice rings.** Distance-free test: observed 1245/847 = 1.4699 vs
  predicted tan-ratio (110)/(100) = 1.4712 — agreement **−0.09 %**. Per-ring effective
  distance: D_eff(100→847 px) = 196.99 mm, D_eff(110→1245 px) = 196.81 mm — spread
  0.09 %. Both are needle-sharp in the curve, and masking-06 saw discrete Bragg spots
  on exactly these radii. Assignment: **847 px = (100), 1245 px = (110)**.
- **The inner three do not index.** Under D_eff, q(367/531/734 px) = 0.675 / 0.969 /
  1.322 Å⁻¹ → N = (q/q₁₀₀)² = 0.20 / 0.41 / 0.77 — not integers, and the features are
  broad. They are diffuse instrument/sample-environment scattering, kept in the
  registry as `diffuse_anchors` — the historical "four/five LaB6 rings" label was
  operationally useful but physically loose.
- **Completeness closes.** q(111) = 2.618 Å⁻¹ > q_max ≈ 2.36 (r = 1400 px at D_eff) —
  correctly absent; (100) and (110) are the only reachable reflections, both found.
- **Geometry-constant finding.** step4's nominal D = 190 mm vs D_eff ≈ 196.9 mm
  (−3.5 %): the q-axis labels on iq.png are ~3.4 % high (847 px is labeled q = 1.563
  but is truly 1.5115). Exactly the `geometry_constant_mismatch` escalation this check
  exists to raise; px-space anchors are unaffected.
- **The generic machinery is what does the work.** On trial_06/07 the D-voting path —
  taking only the theoretical q table, no px anchors as input — independently
  re-finds (100)@846 / (110)@1243 (2 votes, D_eff 196.63 mm, spread 0.13 %) and
  PASSes; the validated-anchor layer then activates because the run's D_eff matches
  the anchors' `validated_deff_mm` within 2 %. Synthetic selftest
  (`material_expectation.py --selftest`, D_true 100 mm with nominal deliberately 5 %
  off; AgBh at 1000 mm): identity confirmed and D_true recovered within 0.05 % for
  LaB6 (6 rings), Si (2), CeO2 (3), Al2O3 (6), AgBh (4 orders) — same code, only the
  registry row changes.

## When to use

Calibration runs and any run where the sample identity is known and registered.
Complements — does not replace — the manifest peak-window checks used for novel or
unknown samples ([03a](03a_peak_centroid_window.md)/[03b](03b_peak_shape_fit.md)).

## Trade-offs

- **Relative intensities are only coarse witnesses**: texture, large-grain spottiness,
  and partial-arc coverage (the 1245 px ring's arc covers only a few % of azimuth)
  distort them — use per-ring contrast floors, not structure-factor ratios.
- **Registry curation is a human act**: lattice constants are temperature-dependent at
  the 10⁻⁴ level (below tolerance here, but state the reference), and every new
  beamline geometry needs its anchors re-validated before they gate anything.
- **Flat-detector tan model**: detector tilt folds into the D_eff spread rather than
  being fitted — a real tilt shows up as rule-6 evidence, and the drift check's
  residual-pattern table does the diagnosis.
- Registry-only materials (Si, CeO2, AgBh, Al2O3) carry full theoretical q tables but
  no validated anchors/contrasts yet — they gate identity, positions, and
  completeness through the same D-voting machinery as LaB6, not strengths.
- **Sparse-reflection geometries degrade gracefully**: a material with only one
  reachable reflection cannot vote (identity underdetermined, soft) — prefer a
  calibrant with ≥ 2 reachable rings for the beamline's q window (at this setup's
  q_max ≈ 2.36: LaB6 gives 2, CeO2 gives 2, Si only 1, AgBh many at low q).

## Do not

- **Do not tune L, beam center, or λ to make predicted rings land** — the load-bearing
  rule of every layer of this pipeline. D_eff is a *witness*, not a fit parameter to
  silently adopt.
- Do not add, drop, or relabel a registry entry (ring ↔ anchor) mid-run to pass a
  check — registry edits are human-reviewed, backed by an indexing analysis like the
  one above.
- Do not cap the enumeration or search range (golden rule 5).

## Outputs

Contributes to `qa_report.json` under `material_expectation`:
`{material, q_range_checked_invA, lattice_rings: [{hkl, q_pred, r_pred_px, found,
r_obs_px, delta_px, contrast, fwhm_px, D_eff_mm}], ratio_test: {observed, predicted,
diff_pct}, deff: {per_ring, spread_pct, nominal_mm, vs_nominal_pct},
diffuse_anchors: [...], extra_sharp_features: [...], escalations, verdict}`.
Feasibility reports: `outputs/qa_material_expectation_feasibility/`.

## Machine block

```json
{
  "id": "known_material_expectation",
  "anchor_delta_px_max": 5,
  "ratio_tol_pct": 0.5,
  "deff_spread_pct_max": 0.5,
  "deff_vs_nominal_warn_pct": 1.0,
  "sharp_fwhm_max_px": 15,
  "extra_feature_prominence_sigma": 6,
  "completeness_edge_margin_pct": 0.5,
  "anchor_geometry_tol_pct": 2.0,
  "d_vote_bounds_x_nominal": [0.5, 2.0],
  "lambda_A_default": 1.2915,
  "pixel_mm": 0.075,
  "materials": {
    "LaB6": {
      "system": "cubic", "centering": "P", "a_A": 4.15695,
      "q_hkl_invA": {"100": 1.5115, "110": 2.1376, "111": 2.6180, "200": 3.0230,
                     "210": 3.3798, "211": 3.7024, "220": 4.2751, "300": 4.5345,
                     "310": 4.7797},
      "validated_deff_mm": 196.9,
      "lattice_rings_validated": {
        "100": {"r_px": 847, "contrast_min": 1.0},
        "110": {"r_px": 1245, "contrast_min": 2.0}
      },
      "diffuse_anchors_validated": {
        "367": {"contrast_min": 0.30},
        "531": {"contrast_min": 0.010},
        "734": {"contrast_min": 0.045}
      },
      "status": "validated (Run0475, D_eff 196.9 mm, 2026-07-28)"
    },
    "Si": {
      "system": "cubic", "centering": "diamond", "a_A": 5.43102,
      "q_hkl_invA": {"111": 2.0038, "220": 3.2722, "311": 3.8370, "400": 4.6276,
                     "331": 5.0428},
      "status": "registry-only"
    },
    "CeO2": {
      "system": "cubic", "centering": "F", "a_A": 5.41165,
      "q_hkl_invA": {"111": 2.0110, "200": 2.3221, "220": 3.2839, "311": 3.8508,
                     "222": 4.0220, "400": 4.6442},
      "status": "registry-only"
    },
    "AgBh": {
      "system": "lamellar", "d001_A": 58.380,
      "q_hkl_invA": {"001": 0.1076, "002": 0.2153, "003": 0.3229, "004": 0.4305,
                     "005": 0.5381, "006": 0.6458, "007": 0.7534, "008": 0.8610,
                     "009": 0.9686, "0010": 1.0763},
      "status": "registry-only"
    },
    "Al2O3": {
      "system": "hexagonal", "a_A": 4.7591, "c_A": 12.9918,
      "q_hkl_invA": {"012": 1.8054, "104": 2.4630, "110": 2.6405, "006": 2.9018,
                     "113": 3.0128, "024": 3.6109, "116": 3.9233, "300": 4.5735},
      "status": "registry-only"
    }
  }
}
```

## Links

Part of: [qa](../README.md). Consumes labels from
[02_attr_feature_width](02_attr_feature_width.md); hands geometry evidence to
[06_calib_lab6_drift](06_calib_lab6_drift.md) (the q-scale hard authority); px-space
anchor contract shared with
[verification/01](../../verification/01_iq_quality.md) C1/C2. Range rule:
golden rule 5 in [skills/README](../../README.md) and
[verification/README](../../verification/README.md).
