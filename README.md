# xppl1016922 — local copy + automatic detector masking

Recovered local copy of LCLS experiment **`xppl1016922`** (XPP) and its
Jungfrau1M masking project. The verified notebook reference is available only
for run 475; run 389 raw XTC is incomplete.

## Layout

```
automask/          the project — an installable Python package (import automask)
  io/              reusable readers (small-data + calib + XTC); no project logic
  stats/ regularization/ combine/   the masking method registries
  producers/       build the frozen inputs (need psana, run once)
  studies/         exploratory scripts + the Hydra sweep driver
  conf/ scripts/   Hydra configs + sweep launchers
docs/              experiment + psana background
psana_env.sh       activate the ana-4.0.62 conda env (for XTC / psana)
calib/ xtc/ hdf5/  the data mirror (gitignored — large)
xpp_sharing/       the lab's current production method (read-only baseline)
```

## Setup

```bash
pip install -e .                     # installs declared dependencies from pyproject.toml
# In the psana environment, protect its pinned numerical stack instead:
source psana_env.sh
pip install -e . --no-deps
```

## Run the production pipeline on a given run

```python
python -m automask.studies.sweep_hyperparameters \
    experiment=production combine=union \
    eval.runs=[389,475] eval.synthetic=false figures=true
```

## Read the experiment (`automask.io`)

| module | what it does |
|--------|--------------|
| `automask.io.lcls_xpp` | small-data HDF5 + calib files as plain numpy (`SmallData`, `load_calib`, `resolve`). No psana. |
| `automask.io.read_xtc` | pull calibrated per-event Jungfrau frames from raw XTC (needs psana). |
| `automask.io.setup_psdm_layout` | build the psana-readable `SIT_PSDM_DATA` tree (real-colon calib names). |

```python
from automask.io.lcls_xpp import SmallData

sd  = SmallData(475)
img = sd.sum_image()             # (1030,1064) calibrated run-sum image
geo = sd.jungfrau_geometry()     # distance, wavelength, beam center, masks, ...
```

## Build and run

```bash
# Verified run-475 notebook reference; needs complete run-475 XTC.
python -m automask.producers.baseline_mask --run 475
python -m automask.producers.extract_dataset
python -m automask.producers.build_features

python -m automask.masking
python -m automask.synthetic.evaluate \
  --config automask/synthetic/config/synthetic_pipeline_smoke.yaml
```

See [`automask/README.md`](automask/README.md) for masking, sweeps, and known
recovery limitations; [`docs/`](docs/) covers the local data and XTC access.
