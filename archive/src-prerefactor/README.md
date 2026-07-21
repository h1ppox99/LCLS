# src/ — code for xppl1016922

Two parts:

- **`io/`** — reusable readers for the experiment data (no project logic).
- **`automask/`** — the automated-detector-masking project (the actual research).

Background docs: **[`docs/DATA_OVERVIEW.md`](docs/DATA_OVERVIEW.md)** (the experiment, every
data file, detector geometry) and **[`docs/PSANA_XTC.md`](docs/PSANA_XTC.md)** (reading raw XTC).

## Setup
```bash
pip install -r requirements.txt      # numpy, h5py, scipy, scikit-image, matplotlib
source psana_env.sh                  # only for XTC / psana (activates ana-4.0.62)
```

## io/ — read the experiment
| file | what it does |
|------|--------------|
| `lcls_xpp.py` | small-data HDF5 + calib files as plain numpy (`SmallData`, `load_calib`, `resolve`). No psana. |
| `read_xtc.py` | pull calibrated per-event Jungfrau frames from raw XTC (needs psana). |
| `setup_psdm_layout.py` | build the psana-readable `SIT_PSDM_DATA` tree (real-colon calib names). |

```python
import sys; sys.path.insert(0, "io")
from lcls_xpp import SmallData

sd  = SmallData(475)
img = sd.sum_image()             # (1030,1064) calibrated run-sum image
geo = sd.jungfrau_geometry()     # distance, wavelength, beam center, masks, ...
```

## automask/ — the project
Automate Jungfrau1M detector masking to replace the lab's manual hand-drawn mask. After a
one-time extract, the masking code is numpy-only. Full details in
[`automask/README.md`](automask/README.md). Structure:

- `dataset.py`, `methods.py` — the numpy-only loader + the masking pipeline.
- `producers/` — build automask's frozen inputs (need psana/h5py, run once):
  `extract_dataset.py`, `baseline_mask.py` (the baseline mask), `normalized_median.py`,
  `band_features.py`.
- `studies/` — exploratory / diagnostic scripts (not part of the pipeline).
- `data/` — frozen numpy inputs (images, masks, features).
- `outputs/` — everything generated (`figures/`, `masks/`, `cache/`).

## Notes
- Only runs **389** and **475** have small-data; run **475** has full XTC (all streams),
  run **389** only a partial stream — see the repo `CLAUDE.md`.
- Use the **190 mm** sample–detector distance, not the per-pixel `z` map (stale 100 mm).
- `psana_env.sh` stays at `src/` root — `source src/psana_env.sh`.
