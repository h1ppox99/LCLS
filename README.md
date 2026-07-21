# xppl1016922 — local copy + automatic detector masking

A local copy of LCLS experiment **`xppl1016922`** (XPP instrument, SLAC), plus the research
project built on top of it: an **automatic, run-agnostic masker for the Jungfrau1M detector**
to replace the lab's manual hand-drawn mask. Working guide for the codebase: **[`CLAUDE.md`](CLAUDE.md)**.
Background docs: **[`docs/DATA_OVERVIEW.md`](docs/DATA_OVERVIEW.md)** (the experiment, every
data file, detector geometry) and **[`docs/PSANA_XTC.md`](docs/PSANA_XTC.md)** (reading raw XTC).

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
source psana_env.sh                  # only for XTC / psana (activates ana-4.0.62)
pip install -e . --no-deps           # make `import automask` work everywhere
# --no-deps protects the ana env; for a plain numpy env instead:
# pip install -r requirements.txt
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

## Run the masker

```bash
python -m automask.masking                 # production report (IoU vs the hand mask)
bash automask/scripts/eval_production.sh   # the Hydra regression anchor
```

Full details in [`automask/README.md`](automask/README.md).
