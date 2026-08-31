# Baseline recipe

The robust, low-variance default. Every masking task starts here and changes it
one attributable concern at a time (see [evidence-and-decisions.md](evidence-and-decisions.md)).
Do not rebuild a recipe from scratch. Confirm current defaults with `automask_catalog`; the
values below mirror `production_pipeline()` in `automask/masking.py`.

## Default shot selection

Field names are run-specific — copy the exact incident-monitor name from the
inspection `report.md`, never hardcode it. Provenance: `experiment_logs/shot_selection_report.md`.

- **Beam-on gate** — `eventCode[137] == 1` (drops ~1% no-beam shots).
- **Low-flux floor** — a low-side cut on an upstream incident monitor
  (`gas_detector/f_11_ENRC`, else `ipm2/sum`) at ~0.5·median. The darkest ~10%
  are structureless and only add noise. 
- **No per-shot normalization** — `normalization=None`. Normalizing erases the
  persistent structure masking depends on.
- **No trim** by default; add a small low-side trim only if the incident
  distribution shows a bad tail. Never trim the high side.
- **`n_shots ≈ 800`** — the default sample. 400 minimum; ~1600 for sharp
  features, ~400 for diffuse; flag results below ~200 shots as weak.
- **Reductions** — `mean` + `std` feed the default channels; `mad` is available
  for the instability floor (see Known limits).

## Default pipeline

One channel list unioned onto the intensity-free floor. Floor channels are
mandatory and preserved in every candidate.

| channel | kind | key params | field_reg | mask_reg | catches |
|---|---|---|---|---|---|
| `geometry` | floor | pad=2, frac=0.4 | — | — | unmapped canvas + ASIC/gap borders (100% precision) |
| `status_as_mask` | floor | pad=2 | — | — | psana bad-pixel status, dilated |
| `variance` | field, low | k=4, mode=low | tv(weight=4.0) | — | dead / shadowed pixels (low per-pixel std) |
| `hough_lines` | pick | defaults | none | — | straight dark lines — shadows, scratches, ASIC seams |
| `asic_polish` | field, high | asic=256, n_iter=3, k=15.0 | blob_scale(6,9,12,16) | fill_holes, area_gate(min_area=200) | extended / circular pedestal defects psana status misses |

`Sample.needs()` for this recipe: `mean`, `std`, `pedestals`, `status_as_mask`,
`real`. `hough_lines` emits its own boolean, so its `field_reg` must be `None`.

## Known limits of the baseline

Where the default is expected to fall short — the starting map for adaptation:

- **Instability floor is not on by default.** Persistently unstable pixels
  (jitter without a mean/std signature) need a `mad_variance` channel on the
  `mad` reduction. Add it for runs where fold consistency flags unstable pixels.
- **`hough_lines` is high-precision by design** — it fires only on genuine
  straight lines and deliberately ignores faint or curved ones.
- **`variance` (low) can over-reach onto real dark structure** — genuine
  shadows and low-intensity halo look dead. Relate its layer to the selection
  image before trusting it.
- **No azimuthal channel by default.** Ring-anomaly detection (`sigma_clipping`)
  needs the beam center and must be added explicitly.
- **`asic_polish` works in the pedestal domain** — it will not catch
  intensity-domain artifacts that only appear in the selected-shot image.
- **The low-flux threshold and `n_shots` are starting candidates**, not fixed
  rules; branch (CC/VCC) is irrelevant to masking.
