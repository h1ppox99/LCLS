# xppl1016922 — local copy + automatic detector masking

Recovered local copy of LCLS experiment **`xppl1016922`** (XPP) and its
Jungfrau1M masking project. The verified notebook reference is available only
for run 475; run 389 raw XTC is incomplete.

## Layout

```
automask/          the project — an installable Python package (import automask)
  io/              reusable readers (small-data + calib + XTC); no project logic
  stats/ regularization/ combine/   the masking method registries
  features/        per-run feature specs + cached FeatureStore
  producers/       build the frozen inputs (need psana, run once)
  studies/         exploratory scripts + the Hydra sweep driver
  synthetic/       synthetic-artifact benchmark (no real data needed)
  conf/ scripts/   Hydra configs + sweep launchers
docs/              experiment + psana background (DATA_OVERVIEW, PSANA_XTC, DATA)
psana_env.sh       activate the ana-4.0.62 conda env (for XTC / psana)
calib/ xtc/ hdf5/  the data mirror (gitignored — large)
xpp_sharing/       the lab's current production method (read-only baseline)
```

## Setup

### First time setup

The XTC readers need psana, which is intentionally not installed by this
project's `pip` dependencies. Choose an environment path and set the same path
as `ENVP` in `psana_env.sh`.

```bash
ENVP=/Data/$USER/envs/ana-4.0.62

conda create -p "$ENVP" \
  -c lcls-i -c conda-forge \
  psana=4.0.62 python=3.9 numpy h5py
```

Edit `psana_env.sh` so its `ENVP` value matches the path above, then install
this project from inside that environment. `--no-deps` preserves psana's pinned
scientific packages.

```bash
source psana_env.sh
python -m pip install -e . --no-deps
```

If an import reports a missing non-psana dependency, install it into this Conda
environment (for example, `conda install -p "$ENVP" -c conda-forge hydra-core`).

### Everyday setup

```bash
source psana_env.sh
```

For workflows that do not read XTC data or import psana, a normal Python
environment can instead use `python -m pip install -e .`.

## Run the production pipeline 

```python
 python -m automask.masking
```

## Read the experiment (`automask.io`)

| module | what it does |
|--------|--------------|
| `automask.io.lcls_xpp` | small-data HDF5 as plain numpy (`SmallData`, `resolve`). No psana. |
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
