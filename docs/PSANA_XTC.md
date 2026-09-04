# Reading raw XTC with psana

Raw XTC is the production source for profiling and selected-shot reductions.
The repository supports one code path with two data-source backends; neither
backend requires data symlinks inside the checkout.

## Select a backend

| backend | intended location | run access | calibration |
| --- | --- | --- | --- |
| `slac` | LCLS/S3DF | standard experiment resolver | facility store |
| `local` | copied data elsewhere | explicit files in `AUTOMASK_XTC_DIR` | `AUTOMASK_CALIB_DIR` |

Use `scripts/create_environment.sh` as shown in the root README, or configure an
existing environment by copying `psana_env.local.example` to
`psana_env.local`. The local form is:

```bash
PSANA_ENV=/path/to/software/envs/automask-py311
PSANA_CONDA_SH=/path/to/miniforge/etc/profile.d/conda.sh
SMALLDATA_TOOLS=/path/to/software/src/smalldata_tools
AUTOMASK_BACKEND=local
AUTOMASK_XTC_DIR=/path/to/xppl1016922/xtc
AUTOMASK_CALIB_DIR=/path/to/xppl1016922/calib
AUTOMASK_CACHE_DIR=/path/to/writable/cache
```

For SLAC, first source the standard LCLS psana setup, then set
`AUTOMASK_BACKEND=slac` and omit the three local data paths. The facility
`SIT_PSDM_DATA`, `SIT_ROOT`, and `SIT_DATA` values are preserved and validated.
For local psana1 imports, activation sets the minimum `SIT_ROOT` and
`SIT_PSDM_DATA` values to the real data root; it does not create an experiment
registry or a fake PSDM directory.

Activate and open a run through the common resolver:

```bash
source psana_env.sh
```

```python
from automask.io.read_xtc import run_source

source = run_source(475)
print(source.backend)
```

Pass `backend="local"` or `backend="slac"` when a call must override the
environment. `ImageStore`, run profiling, geometry, and calibration all retain
the resolved source, so one operation cannot accidentally mix local geometry
with SLAC calibration.

## Local data currently configured on the development machine

The external data remains at `/home/groups/darve/hippowal/LCLS`; these paths are
not part of a clone.

| run | streams on this machine | state |
| --- | --- | --- |
| 475 | s00–s04 | complete, 3,201 events |
| 389 | s03 | truncated |
| 378 | s00 | partial |
| 396 | s00 | partial |

Partial and truncated local runs work because the local backend passes the
actual stream files to psana. Never join run-389 XTC events to small-data rows by
index.

## Calibration names

The local calibration tree has been normalized for Linux and now uses literal
colons, for example:

```text
Jungfrau::CalibV1/XppEndstation.0:Jungfrau.0
```

The former macOS private-use replacements and Finder metadata were removed.
Do not translate colons in new code.

## Verify a setup

```bash
source psana_env.sh
lcls-agent doctor
python - <<'PY'
from automask.io.read_xtc import detector_calibration, panel_geometry, run_source

source = run_source(475)
print(source.backend)
print(detector_calibration(475, "pedestals", source=source).shape)
print([array.shape for array in panel_geometry(475, source=source)])
PY
```

The expected Jungfrau panel shape is `(2, 512, 1024)`. A local calibration
failure must be treated as a setup error; raw frames alone are not enough for
the production pipeline.
