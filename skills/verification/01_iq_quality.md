---
name: xray-verification-iq-quality
description: First-line acceptance criterion for the pipeline endpoint — LaB6 ring presence and contrast in I(q), background sanity (no negative bins, bounded noise/bumps), and mask-coverage bounds. Thresholds calibrated on the validated Run0475 endpoints (agent_trial_02/03); machine-readable block at the bottom.
---

# Verification · Criterion 1 — I(q) quality (rings + background + coverage)

**Family:** endpoint acceptance. **Part of:** [verification](README.md).
**Inputs:** `iq_metrics.json` (from `pipeline/step4_iq.py`), `iq.png` for a visual sanity look.

## What is checked, and why

### C1 — Ring presence & position
All four LaB6 rings visible on this detector must be found within **±5 px** of their
expected radii `[367, 531, 734, 847]`. A missing or shifted ring means the sum is wrong
at a structural level: bad event alignment, wrong calibration, or a beam-center error —
usually a **reduction**-side failure (or a geometry regression).

### C2 — Ring contrast
Per-ring local contrast `(I_peak − I_bg,local) / I_bg,local` must exceed a per-ring
minimum. Contrast collapses when shots are summed with wrong weights/normalization
(washes out rings) or when low-signal shots pollute the sum → **reduction**.
The rings differ hugely in intrinsic strength, so thresholds are per-ring —
validated values on good runs: 0.65 / 0.02–0.03 / 0.10 / 2.0. Thresholds are set at
roughly half the validated values:

| ring (px) | validated contrast | minimum required |
|---|---|---|
| 367 | 0.65 | **0.30** |
| 531 | 0.02–0.03 | **0.010** |
| 734 | 0.10 | **0.045** |
| 847 | 2.0 | **1.0** |

### C3 — Background sanity
Computed over the ring-free windows `[150–330, 410–500, 570–700, 770–820, 880–1000] px`:

- `neg_bin_fraction` ≤ **0.02** — a summed, normalized powder background cannot be
  negative; negative bins mean unmasked shadow/dead regions leak into the average
  (validated runs: 0.0) → **mask**.
- `rel_noise` (MAD of detrended background / background level) ≤ **0.03** — excess
  bin-to-bin scatter means unmasked hot/defect pixels (validated: 0.012) → **mask**.
- `bump_max_sigma` ≤ **10** — the largest detrended excursion in ring-free windows;
  a strong localized bump is an unmasked parasitic feature (validated: 4.3–4.7,
  reflecting ordinary weak structure; a real unmasked artifact drives it far higher)
  → **mask**.

### C4 — Coverage bounds
- `panel_mask_fraction` ≤ **0.25** — over-masking destroys statistics (validated 0.06–0.11).
  Over-masking is a **mask**-side failure.
- `valid_bins` ≥ **1200** of 1405 — the radial range must stay usable (validated 1335–1352).

## Verdict & feedback routing

FAIL any sub-check ⇒ criterion fails ⇒ run fails. Route feedback:
C1/C2 → `reduction` (unless the mask obviously ate the rings — check `panel_mask_fraction`
first); C3/C4 → `mask`. Always quote the measured number, the threshold, and a concrete
suggested change.

## Machine thresholds

```json
{
  "id": "iq_quality",
  "rings_expected_px": [367, 531, 734, 847],
  "ring_delta_px_max": 5,
  "ring_contrast_min": {"367": 0.30, "531": 0.010, "734": 0.045, "847": 1.0},
  "neg_bin_fraction_max": 0.02,
  "rel_noise_max": 0.03,
  "bump_max_sigma_max": 10.0,
  "panel_mask_fraction_max": 0.25,
  "valid_bins_min": 1200
}
```
