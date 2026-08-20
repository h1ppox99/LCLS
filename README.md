# xppl1016922 — local copy + automatic detector masking

Recovered local copy of LCLS experiment **`xppl1016922`** (XPP) and its
Jungfrau1M masking project. The verified notebook reference is available only
for run 475; run 389 raw XTC is incomplete.

## Layout

```
automask/          the project — an installable Python package (import automask)
  io/              production psana/XTC readers and official detector adapters
  dev/             local-mirror setup and raw-format validation tools
  stats/ regularization/   the masking method registries (combine/ is the fixed union)
  image_store.py   selected-shot reductions + detector calibration cache
  producers/       optionally prewarm image caches from psana/XTC
  studies/         exploratory scripts + the Hydra sweep driver
  synthetic/       synthetic-artifact benchmark (no real data needed)
  conf/ scripts/   Hydra configs + sweep launchers
docs/              experiment + psana background (DATA_OVERVIEW, PSANA_XTC, DATA)
psana_env.sh       activate the ana-4.0.66-py311 conda env (for XTC / psana)
calib/ xtc/        the production data mirror (gitignored — large)
xpp_sharing/       the lab's current production method (read-only baseline)
```

## Setup

### First time setup

The XTC readers need psana, which is intentionally not installed by this
project's `pip` dependencies. Choose an environment path and set the same path
as `PSANA_ENV` in `psana_env.local`.

```bash
ENVP=/Data/$USER/envs/ana-4.0.66-py311

conda create -p "$ENVP" \
  -c lcls-i -c conda-forge \
  psana=4.0.66 python=3.11 numpy h5py pytest
```

Edit `psana_env.local` so its `PSANA_ENV` value matches the path above, then install
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

## Experiment access

| module | what it does |
|--------|--------------|
| `automask.io.read_xtc` | production calibrated Jungfrau access through psana. |
| `automask.io.lcls1_adapters` | official SLAC scalar adapters used on psana events during run profiling. |
| `automask.dev` | local mirror setup and raw-format diagnostics; never deployed at SLAC. |

```python
from automask.utils import profile_run_values
from automask.image_store import ImageStore
from automask.shot_selection import Condition, ShotSelection

profile = profile_run_values(475)
selection = ShotSelection(where=(Condition("ai/ch03", "<=", 2.0),))
event_indices = selection.resolve(profile)
mean_image = ImageStore(run_profile=profile).reduce(475, selection, "mean")
```

## Using automask

`automask` is a library: capability discovery, reusable run-inspection profiles,
field-native shot selection, image reductions, mask construction, and run-local
validation are all called directly from Python.

```python
from automask.catalog import capability_catalog
from automask.run_inspection import inspect_run
```

See [`docs/AUTOMASK.md`](docs/AUTOMASK.md) for the scripting workflow and
conventions. The agent reaches the same operations through the standalone stdio
MCP server in `lcls_agent/`; there is no human-facing command-line tool.
The checked-in `.mcp.json` registers that server for direct Claude Code use;
activate `psana_env.sh` before launching Claude Code and approve the project
server on first use. See [`lcls_agent/README.md`](lcls_agent/README.md).

## Build and run

```bash
# Optional: prewarm production selected-shot images from XTC.
python -m automask.producers.build_images

python -m automask.masking
python -m automask.synthetic.evaluate \
  --config automask/synthetic/config/synthetic_pipeline_smoke.yaml
```

See [`automask/README.md`](automask/README.md) for masking, sweeps, and known
recovery limitations; [`docs/`](docs/) covers the local data and XTC access.
