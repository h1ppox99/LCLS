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
| `research/` | dev/research tooling, off the runtime path (synthetic studies, hand-mask editor) |

## Image cache

The ImageStore cache is intentionally gitignored. Set `AUTOMASK_CACHE_DIR` to
keep it on scratch/group storage. It can be prewarmed or populated on demand:

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

## Reference data

Human masks belong in `reference_masks/` as boolean assembled `.npy` arrays with
`True == masked`, named `reference_mask_run<run>.npy`. Labelled evaluation runs
against whichever runs have a mask there; consistency and perturbation
validation remain available regardless. See `reference_masks/README.md`.

## Dependencies

Runtime dependencies, including the required Claude SDK/MCP host, are declared
in the repository-root `pyproject.toml`. The validated psana/compiled stack is
captured in `environment.yml`; `h5py` is used for bounded temporary staging
during robust reductions.

## Scoring against references

Load a reference with `automask.evaluation.reference_mask(run)` and score a
prediction with `automask.evaluation.dataset.score`; the full
labelled/consistency contract is in
[`docs/EVALUATION.md`](../docs/EVALUATION.md). `reference_masks/` is the only
frozen input left — every array a pipeline consumes is computed from the run.
