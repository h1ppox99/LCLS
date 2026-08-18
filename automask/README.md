# automask — automated detector masking (xppl1016922, Jungfrau1M)

This is the masking project for the recovered LCLS experiment. Production run
profiling, calibration, geometry, and selected-shot image materialization use psana/XTC.

## Image cache

The ImageStore cache is intentionally gitignored. It can be prewarmed, or
populated on demand by the masking pipeline:

```bash
python -m automask.producers.build_images
```

`build_images` is optional: it only prewarms the ImageStore cache, which the
evaluation loop otherwise fills on demand from raw XTC.

## Building a mask

```python
from automask.masking import Channel, Pipeline
from automask.sample import Sample

pipe = Pipeline(
    channels=[
        Channel("geometry", field_reg=None),  # floor: unmapped/ASIC lines
        Channel("status_as_mask", field_reg=None),  # floor: psana pixel status, per run
        Channel("variance", VarianceParams(k=3.5, mode="low"), field_reg="tv"),
    ]
)
sample = Sample.from_store(475, selection, pipe.needs())
mask = pipe.run(sample)  # bool, True == masked
```

One list of channels. Whether a channel belongs to the intensity-free floor is
read from its stat's registered `kind`, so floor channels are configured — and
parameterised — exactly like every other one.

## Adding a method

Drop one file in `stats/` or `regularization/` that defines a compute
function, a `Params` dataclass, and a `register_*` call; add it to that package's
`__init__.py` import line and a matching `conf/<group>/<name>.yaml`. A stat's
`needs` names the Sample arrays it reads — reductions (`mean`/`std`/`median`/`mad`)
or psana calibration accessors (`pedestals`, `rms`, `status_as_mask`). It is then
selectable by name everywhere (`Pipeline`, the registries, and the sweep driver).

## Sweeping hyperparameters

```
# Fit one detector on the tuning runs (writes results.csv):
python -m automask.studies.sweep_hyperparameters -m stat=variance \
    stat.params.k=2,2.5,3,3.5 regularization.params.weight=5,10,15

# Validate the chosen configuration once, without a sweep:
python -m automask.studies.sweep_hyperparameters experiment=production \
    eval.phase=validate
```

`scripts/*.sh` are thin wrappers over the driver reproducing the old per-method sweeps.
Synthetic injection is a separate stress test, not the parameter-selection target.
The complete evaluation contract is in `docs/EVALUATION.md`.

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

## Usage

```python
# automask is an installed package (pip install -e . --no-deps) — import directly:
from automask.dataset import load_mask, score
from automask.image_store import ImageStore
from automask.selection_presets import BEAM_ON_SELECTION

img = ImageStore().reduce(475, BEAM_ON_SELECTION, "mean")  # (1064,1030), from XTC
gt = load_mask("human_Mask")  # bool, True==masked

# ... your auto-masking algorithm ...
pred = img == 0  # trivial baseline

print(score(pred, gt))  # {'iou':.., 'precision':.., 'recall':..}
```

`data/masks/` holds the hand-drawn references and is the only frozen input left;
every array a pipeline consumes is computed from the run.

## Prewarm production images

```
python -m automask.producers.build_images
```

## Baseline to beat

The lab's original notebook workflow remains under `xpp_sharing/` as the
read-only comparison point. Production automasking does not import it.
