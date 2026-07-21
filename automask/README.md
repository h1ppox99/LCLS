function, a `Params` dataclass, and a `register_*` call; add it to that package's
`__init__.py` import line and a matching `conf/<group>/<name>.yaml`. It is then
selectable by name everywhere (`Pipeline`, the registries, and the sweep driver).

## Sweeping hyperparameters

```
# one detector, swept over K x TV weight on both eval runs (writes results.csv):
python -m automask.studies.sweep_hyperparameters -m stat=variance \
    stat.params.k=2,2.5,3,3.5 regularization.params.weight=5,10,15

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
| `human_Mask` | 13.86% | **primary target** — hand-drawn: dead pixels **+** geometry (beam-stop rectangles/triangle) |
| `cmask_run{389,475}` | 1.90% | production combined bad-pixel mask |
| `mask_run{389,475}`  | 1.98% | production bad-pixel mask |
| `statusMask_run{389,475}` | 0.42% | derived from calibration `pixel_status` (pure detector bad pixels) |

Note the two families measure different things: `human_Mask` includes **geometry**
regions; the `*mask*` family is **bad pixels only**. A full auto-masker must
produce both components (bad-pixel detection **+** geometry/zero-region detection).

## Dependencies

`numpy`, `scipy`, `matplotlib`, and `scikit-image` (the last only for
`stats/blackhat.py`'s grey closing), plus `hydra-core` for the sweep driver. See
`../requirements.txt`. Still no `psana` /
`h5py` outside the one-time `extract_dataset.py` step.

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

## Reproduce the frozen data

```
python -m automask.producers.extract_dataset   # re-reads small-data, rewrites data/*.npy + manifest.json
```

## Baseline to beat

`producers/baseline_mask.py` is the faithful transcription of the notebook's manual
recipe (zero-mask via `sumimg<=0` + 5×5 dilation, plus 3 hand-drawn rectangles and a