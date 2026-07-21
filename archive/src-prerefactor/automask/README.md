# automask — automated detector masking (xppl1016922, Jungfrau1M)

Self-contained research project for **building/automating detector-mask creation**.
It is deliberately decoupled from the LCLS analysis pipeline: after a one-time
extraction step, everything here runs with **numpy only** — no `psana`, no
`h5py`, no LCLS filesystem.

## Goal

Reproduce and then improve on the hand-drawn mask (`Mask.npy`) that
`xpp_sharing/2_Lab6_Mask_calibration.ipynb` builds manually — i.e. automate the
"find zero/dead regions + exclude geometry regions" step from a **sum image**.

## Layout

```
automask/
├── dataset.py           # numpy-only loader: load_image / load_mask / score
├── masking.py           # THE library: STATS/REGULARIZERS/COMBINERS registries,
│                         # Detector + Pipeline runner, production_pipeline(), mask_image().
│                         # Production (run 475): floor IoU 0.502 prec 1.00; union combo
│                         # IoU 0.732; weighted-sum combo_sum IoU 0.747.
├── evaluation.py        # Sample context + load_sample(run) + evaluate(pipeline, runs);
│                         # EVAL_RUNS is the single place the eval set grows.
├── viz.py               # shared figures: agree_rgb, agreement maps, results heatmap
├── stats/               # one file per statistic (compute fn + Params dataclass)
│   ├── variance.py  window_median.py  blackhat.py     # the three production detectors
│   ├── radial_median.py  azimuthal_sigma.py           # promoted candidate detectors
│   └── geometry.py  calib.py                          # intensity-free floor (no sweep)
├── regularization/      # tv.py (field) · pad.py, close_open.py (mask)
├── combine/             # union.py (picks) · weighted_sum.py, mahalanobis.py (fields)
├── conf/                # Hydra config groups: stat/ regularization/ mask_reg/ combine/
│                         # eval/ experiment/ (+ config.yaml). Override on the CLI / -m sweep.
├── scripts/             # thin .sh wrappers over the sweep driver (sweep_variance.sh, …)
├── ARCHIVE.md           # discarded masking ideas, one line each
├── producers/           # build the frozen inputs (need psana/h5py, run once)
│   ├── extract_dataset.py    # small-data HDF5 -> frozen data/*.npy + manifest
│   ├── baseline_mask.py      # the lab's manual mask, reproduced (the baseline to beat)
│   ├── normalized_median.py  # per-shot-normalized robust reference (umean/ustd features)
│   └── band_features.py      # per-pixel mean/std features for one intensity band
├── studies/             # exploratory / diagnostic (not part of the pipeline)
│   ├── sweep_hyperparameters.py   # the ONE general Hydra sweep driver
│   ├── bin_intensity.py  pixel_stats.py  make_reference_figures.py
│   └── archive/               # pre-refactor scripts on the removed methods API (see its README)
├── data/
│   ├── images/          # calibrated run-sum detector images  (input)
│   ├── masks/           # reference masks (ground truth / targets)
│   ├── features/        # per-pixel mean/umean/ustd feature maps
│   └── manifest.json    # provenance + shapes + masked fractions
└── outputs/             # everything generated
    ├── figures/         #   .png figures
    ├── masks/           #   generated mask .npy
    ├── sweeps/          #   Hydra run/multirun outputs (results.csv per sweep)
    └── cache/           #   heavy intermediate HDF5 / npz
```

## Adding a method

Drop one file in `stats/`, `regularization/`, or `combine/` that defines a compute
function, a `Params` dataclass, and a `register_*` call; add it to that package's
`__init__.py` import line and a matching `conf/<group>/<name>.yaml`. It is then
selectable by name everywhere (`Pipeline`, the registries, and the sweep driver).

## Sweeping hyperparameters

```
# one detector, swept over K x TV weight on both eval runs (writes results.csv):
python studies/sweep_hyperparameters.py -m stat=variance \
    stat.params.k=2,2.5,3,3.5 regularization.params.weight=5,10,15

# the live production recipe (regression anchor):
python studies/sweep_hyperparameters.py experiment=production            # union combo
python studies/sweep_hyperparameters.py experiment=production combine=weighted_sum
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
import sys; sys.path.insert(0, ".")   # from src/automask/
from dataset import load_image, load_mask, score

img = load_image("sum_calib_run0475")     # (1064,1030) float32
gt  = load_mask("human_Mask")             # (1064,1030) bool, True==masked

# ... your auto-masking algorithm ...
pred = img == 0                            # trivial baseline

print(score(pred, gt))                     # {'iou':.., 'precision':.., 'recall':..}
```

## Reproduce the frozen data

```
python producers/extract_dataset.py   # re-reads small-data, rewrites data/*.npy + manifest.json
```

## Baseline to beat

`producers/baseline_mask.py` is the faithful transcription of the notebook's manual
recipe (zero-mask via `sumimg<=0` + 5×5 dilation, plus 3 hand-drawn rectangles and a
triangle). It reproduces `human_Mask` (saved as `data/masks/human_Mask_source.npy`).
That hand-tuned geometry is exactly what this project aims to replace with something
automatic and run-agnostic.
