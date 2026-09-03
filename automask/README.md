# automask — automated detector masking (xppl1016922, Jungfrau1M)

This is the masking project for the recovered LCLS experiment. Production run
profiling, calibration, geometry, and selected-shot image materialization use psana/XTC.

## Layout

The package mirrors the four masking stages plus two seams:

| dir | role |
|---|---|
| `profiling/` | stage 1 — one psana pass → per-shot `RunProfile` (+ disk cache, run report) |
| `selection/` | stage 2 — field-native `ShotSelection` and presets |
| `sample/` | the arrays a pipeline reads: `Sample`, the image cache, geometry |
| `mask/` | stage 3 — the `Pipeline`/`Channel` model + `stats/` `regularization/` `combine/` registries |
| `evaluation/` | stage 4 — labelled scoring, consistency, validation, reference masks |
| `io/` | the psana seam (XTC readers, calibration) |
| `interface/` | the agent surface: capability `catalog` + dict⇄object `recipes` |
| `viz.py` | figures |
| `research/` | dev/research tooling, off the runtime path (psdm setup, synthetic, hand-mask editor) |

## Image cache

The ImageStore cache is intentionally gitignored. It can be prewarmed, or
populated on demand by the masking pipeline:

```bash
python -m automask.producers.build_images
```

`build_images` is optional: it only prewarms the ImageStore cache, which the
evaluation loop otherwise fills on demand from raw XTC.

## Building a mask

The end-to-end scripting workflow — inspect a run, select shots, build the mask,
validate it — is in **[`docs/AUTOMASK.md`](../docs/AUTOMASK.md)**, the single
usage guide. A pipeline is one list of `Channel`s; whether a channel belongs to
the intensity-free floor is read from its stat's registered `kind`, so floor
channels are configured exactly like every other one.

## Adding a method

Drop one file in `mask/stats/` or `mask/regularization/` that defines a compute
function, a `Params` dataclass, and a `register_*` call, then add it to that
package's `__init__.py` import line. A stat's `needs` names the Sample arrays it
reads — reductions (`mean`/`std`/`mad`) or psana calibration accessors
(`pedestals`, `rms`, `status_as_mask`). It is then selectable by name everywhere
(`Pipeline` and the registries). The evaluation contract is in
`docs/EVALUATION.md`.

## Data (frozen, numpy-only)

All arrays come in two forms: **`_asm`** = assembled image `(1064, 1030)` (what
`Mask.npy` is), and **`_panel`** = raw Jungfrau geometry `(2, 512, 1024)`.
All masks are **bool with `True == masked (excluded)`**.

### Reference masks (`data/masks/`)
| name | %masked (asm) | what it is |
|---|---|---|
| `human_Mask` | 13.86% | **run-475 target only** — notebook dead-pixel + geometry mask |
| `cmask_run{389,475}` | 1.90% | production combined bad-pixel mask |
| `mask_run{389,475}`  | 1.98% | production bad-pixel mask |
| `statusMask_run{389,475}` | 0.42% | historical freeze of `pixel_status`; the pipeline now reads this from psana per run (verified bit-identical) |

Note the two families measure different things: `human_Mask` includes **geometry**
regions; the `*mask*` family is **bad pixels only**. A full auto-masker must
produce both components (bad-pixel detection **+** geometry/zero-region detection).
There is no verified run-389 hand mask; evaluation currently falls back to the
shared run-475 target, so run-389 real-mask scores are provisional.

## Dependencies

Runtime dependencies are declared in the repository-root `pyproject.toml`.
`psana` remains external and is needed for production XTC access; `h5py` is
used only for bounded temporary staging during robust reductions.

## Scoring against references

Load a reference with `automask.evaluation.dataset.load_mask` and score a
prediction with `score`; the full labelled/consistency contract is in
[`docs/EVALUATION.md`](../docs/EVALUATION.md). `data/masks/` is the only frozen
input left — every array a pipeline consumes is computed from the run.
