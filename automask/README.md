# automask — automated detector masking (xppl1016922, Jungfrau1M)

This is the self-contained masking project for the recovered LCLS experiment.
After the one-time build, masking and sweeps use only the regenerated NumPy
arrays. Raw XTC is needed only to recreate the verified run-475 reference.

## Recovery status

The frozen `automask/data/` arrays are intentionally gitignored and may be
missing after recovery.  Recreate them in this order:

```bash
python -m automask.producers.baseline_mask --source xtc --run 475
python -m automask.producers.extract_dataset
python -m automask.producers.build_features
python -m automask.producers.normalized_median --run 475 --n 800
```

The first command reproduces the original run-475 notebook recipe. The
HDF5-only `--source smalldata` mode is diagnostic only and is not a reference.
`build_features` supplies a fast fallback. Run `normalized_median` afterwards
to replace run-475 `umean`/`ustd` with the robust IPM2-normalized features used
by the lit-beam statistics. It needs complete XTC and roughly 3+ GiB of cache.

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

### Input images — calibrated run sums (`data/images/`)
Per run (389, 475), three sum flavours: `sum_calib`, `sum_calib_dropped`,
`sum_calib_dropped_square` (the last two enable per-pixel variance/RMS if you
want noise-based masking). Source: `Sums/jungfrau1M_alcove_calib*` in small-data
— these are full-run sums (~40k / ~3.2k events), far higher statistics than the
notebook's 100-frame sum.

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
`psana` remains external and is only needed for raw XTC access; `h5py` is used
by the one-time small-data producers.

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

## Rebuild the frozen data

```
python -m automask.producers.baseline_mask --run 475
python -m automask.producers.extract_dataset
python -m automask.producers.build_features
python -m automask.producers.normalized_median --run 475 --n 800
```

## Baseline to beat

`producers/baseline_mask.py` is the faithful transcription of the notebook's manual
recipe (zero-mask via `sumimg<=0` + 5×5 dilation, plus 3 hand-drawn rectangles and a
triangle). It is restored as `automask.producers.baseline_mask`; XTC mode is
the notebook-faithful path and writes `human_Mask_source.npy` for extraction.
