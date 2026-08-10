# automask — automated detector masking (xppl1016922, Jungfrau1M)

This is the masking project for the recovered LCLS experiment. Production run
profiling, calibration, geometry, and feature materialization all use psana/XTC.

## Recovery status

The FeatureStore cache is intentionally gitignored and may be missing after
recovery. It can be prewarmed, or populated on demand by the masking pipeline:

```bash
python -m automask.producers.build_features
```

`build_features` is optional: it only prewarms the FeatureStore cache, which the
evaluation loop otherwise fills on demand from raw XTC.

## Adding a method

Drop one file in `stats/`, `regularization/`, or `combine/` that defines a compute
function, a `Params` dataclass, and a `register_*` call; add it to that package's
`__init__.py` import line and a matching `conf/<group>/<name>.yaml`. It is then
selectable by name everywhere (`Pipeline`, the registries, and the sweep driver).

## Sweeping hyperparameters

```
# one detector, swept over K x TV weight (writes results.csv):
python -m automask.studies.sweep_hyperparameters -m stat=variance \
    stat.params.k=2,2.5,3,3.5 regularization.params.weight=5,10,15 \
    eval.synthetic=false

# the live production recipe (regression anchor):
python -m automask.studies.sweep_hyperparameters experiment=production            # union combo
python -m automask.studies.sweep_hyperparameters experiment=production combine=weighted_sum
```

`scripts/*.sh` are thin wrappers over the driver reproducing the old per-method sweeps.

## Data (frozen, numpy-only)

All arrays come in two forms: **`_asm`** = assembled image `(1064, 1030)` (what
`Mask.npy` is), and **`_panel`** = raw Jungfrau geometry `(2, 512, 1024)`.
All masks are **bool with `True == masked (excluded)`**.

### Input images — archived benchmark arrays (`data/images/`)
The optional real-data evaluation uses frozen run sums if they are available.
They are not inputs to production feature extraction and are no longer rebuilt
by this package.

### Reference masks (`data/masks/`)
| name | %masked (asm) | what it is |
|---|---|---|
| `human_Mask` | 13.86% | **run-475 target only** — notebook dead-pixel + geometry mask |
| `cmask_run{389,475}` | 1.90% | production combined bad-pixel mask |
| `mask_run{389,475}`  | 1.98% | production bad-pixel mask |
| `statusMask_run{389,475}` | 0.42% | derived from calibration `pixel_status` (pure detector bad pixels) |

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
from automask.dataset import load_image, load_mask, score

img = load_image("sum_calib_run0475")     # (1064,1030) float32
gt  = load_mask("human_Mask")             # (1064,1030) bool, True==masked

# ... your auto-masking algorithm ...
pred = img == 0                            # trivial baseline

print(score(pred, gt))                     # {'iou':.., 'precision':.., 'recall':..}
```

## Prewarm production features

```
python -m automask.producers.build_features
```

## Baseline to beat

The lab's original notebook workflow remains under `xpp_sharing/` as the
read-only comparison point. Production automasking does not import it.
