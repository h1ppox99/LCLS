# Mask design

Stage 3. How a `Pipeline` decides, the operators available, and how to change a
channel deliberately. Start from the baseline pipeline
([baseline.md](baseline.md)); the *trigger* to add or retune a channel, and how
to read its evidence, are in [evidence-and-decisions.md](evidence-and-decisions.md).
Ask `automask_catalog` for the live registry and parameter shapes — do not copy
registry details from memory.

## How a channel decides

A `Channel` is one statistic wrapped in the stages around it. Three stat kinds:

- **field** (`variance`, `mad_variance`, `blackhat`, `asic_polish`,
  `sigma_clipping`) — emits a continuous robust-z field, optionally
  field-regularized (e.g. `tv`), then thresholded at `(k, mode)` to a mask.
  `mode` is `low` (below −k σ), `high` (above +k σ), or `both`.
- **pick** (`hough_lines`) — emits a boolean directly; `field_reg` must be
  `None` and there is no `(k, mode)`.
- **floor** (`geometry`, `status_as_mask`) — intensity-free, 100%-precision
  boolean; always kept.

Evidence channels are gated by `real` (a pixel with no value carries no
evidence); floor channels are not. All picks are unioned onto the floor.

## Invariants

- Masks are boolean, `True == masked`. psana masks are `1 == good` — the
  consuming statistic converts; readers return psana values untouched.
- **Reductions are assembled space; calibration constants are native panel**
  until a stat interprets them. A constant whose zero means something
  (`status_as_mask`) must be read before `geometry.panel_to_asm`.
- **Preserve the `geometry` + `status_as_mask` floor** in every run-level
  candidate — detector gaps/outside area and psana bad-pixel status.
- Detector facts come from psana (panel shape, canvas size, gain-stage count) —
  never reintroduce shape constants.

## Channels and the artifact → operator map

Match the operator to the artifact's morphology, then verify:

| artifact | operator | notes |
|---|---|---|
| dead / shadowed | `variance` (low) | low per-pixel std |
| hot / bad-status | `status_as_mask` (floor) | psana status |
| extended / circular blob | `asic_polish` or `blackhat` + `blob_scale` | pedestal-domain vs image-domain |
| persistently unstable | `mad_variance` | on the `mad` reduction |
| straight dark lines | `hough_lines` | shadows, scratches, ASIC seams |
| ring / azimuthal anomaly | `sigma_clipping` | needs beam center |

**Method:** run the candidate operators via `build_mask`, read each channel's
evidence *outside the floor*, then verify — never accept a layer on pixel count
alone (see [evidence-and-decisions.md](evidence-and-decisions.md)).

## Line detection

`hough_lines` (kind=`pick`) masks straight dark line defects that per-pixel
stats miss. It emits a boolean directly — no `field_reg`/`k`/`mode`; its knobs
(`line_length`, `line_gap`, `bin_k`) act upstream of any threshold.

- Its evidence is a black-hat darkness field with the floor and dead pixels
  inpainted to nearest-live before the morphology — this is what stops the
  status-band edge raising a false straight "rim" line.
- It is **high-precision by design**: it fires only on genuine straight lines and
  deliberately does not chase faint or curved ones.

## Protecting real signal

Bragg peaks and azimuthal rings are legitimate signal; streaks, shadows,
saturated regions, and persistent detector artifacts are mask candidates. **Shape
alone does not distinguish them** — relate every emitted layer to the
selected-shot image, detector geometry, and run context before accepting it. A
hard `area_gate` also helps: single-pixel defects belong to the calibration
mask, so run-specific artifacts are the larger connected components.

## Post-processing (regularizers)

Field regularizers act before the threshold; mask regularizers after. Both slots
accept a list applied left-to-right.

- **`tv`** — denoise a continuous field before thresholding.
- **`blob_scale`** — multi-scale matched filter that aggregates weak coherent
  evidence so a diffuse defect becomes separable (pairs with `asic_polish`).
- **`fill_holes`** — close holes left by thresholding a graded field.
- **`pad`** — dilate sparse picks.
- **`area_gate`** — drop connected components below `min_area` (extended defects
  only); do not use it on a channel meant to catch single pixels.

## Changing a channel deliberately

- Change **one attributable concern at a time**, and define a **new pipeline
  handle** for every alternative — compare, don't mutate.
- Use separate channels for distinct evidence sources or artifact types.
- Inspect the **per-channel layer counts** in the `build_mask` summary as well as
  the combined mask; examine each layer before accepting its contribution.
- Record why each changed parameter is physically or statistically plausible.
  Do not tune solely until the overview looks clean.
