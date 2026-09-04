# xppl1016922 automatic detector masking

This repository contains the installable `automask` package and its required
Claude Agent SDK/MCP host. Raw experiment and calibration data are external and
are never copied, moved, or committed by the setup.

## Setup

The checked-in `environment.yml` pins the validated psana/scientific stack and
the required Claude/MCP dependencies. Run the bootstrap on a login node with
`mamba` or `conda` available, using software storage with enough quota.

`SOFTWARE_ROOT` below is **not** experiment data. It is a writable directory
with enough quota where the bootstrap installs the conda environment
(`SOFTWARE_ROOT/envs/automask-py311`) and clones the pinned `smalldata_tools`
(`SOFTWARE_ROOT/src/smalldata_tools`). On Sherlock put it on `$GROUP_HOME` or
`$SCRATCH`, never `$HOME`.

At SLAC, use the normal facility experiment resolver:

```bash
# First source the standard LCLS psana setup for its SIT_* configuration.
# arg 1: mode = slac (facility resolver, no local paths needed)
# arg 2: SOFTWARE_ROOT = where the env and smalldata_tools are installed
scripts/create_environment.sh slac /path/to/software
source psana_env.sh
```

On another machine, point directly at a local XTC directory and its real-colon
calibration directory:

```bash
# arg 1: mode      = local (open explicit files instead of the resolver)
# arg 2: SOFTWARE  = env + smalldata_tools install dir (needs quota, not data)
# arg 3: XTC_DIR   = holds xppl1016922-r<RUN>-s<STREAM>-c00.xtc files
# arg 4: CALIB_DIR = psana calib store (literal-colon dirs, e.g. Jungfrau.0)
# arg 5: CACHE_DIR = optional writable cache for reductions/profiles
scripts/create_environment.sh local /path/to/software \
    /path/to/xppl1016922/xtc /path/to/xppl1016922/calib \
    /path/to/writable/cache
source psana_env.sh

# e.g. on Sherlock, with the experiment files under $GROUP_HOME/lcls/ and the
# cache pointed at $SCRATCH so it does not eat the 15 GB $HOME quota:
scripts/create_environment.sh local "$GROUP_HOME/lcls/software" \
    "$GROUP_HOME/lcls/xppl1016922/xtc" "$GROUP_HOME/lcls/xppl1016922/calib" \
    "$SCRATCH/lcls_automask_cache"
source psana_env.sh
```

The bootstrap creates the conda environment, checks out the pinned
`smalldata_tools` revision, installs this repository without altering the
compiled stack, and writes the ignored `psana_env.local`. No repository data
symlinks and no synthetic PSDM directory are required. For an existing
environment, copy `psana_env.local.example` and edit the paths instead.

Run the verification after activation:

```bash
lcls-agent doctor
python -m pytest -q automask/tests tests/lcls_agent
```

## Data backends

`AUTOMASK_BACKEND` has two explicit production values:

- `slac` uses `exp=xppl1016922:run=<run>` and the facility calibration store.
- `local` opens `AUTOMASK_XTC_DIR` streams explicitly and uses
  `AUTOMASK_CALIB_DIR`.

`auto` remains a convenience: it uses explicit files when the requested run is
present locally and otherwise uses the SLAC resolver. Scripts and batch jobs
should select `local` or `slac` explicitly.

## Repository map

| path | role |
| --- | --- |
| `automask/` | profiling, selection, masking, evaluation, and psana I/O |
| `lcls_agent/` | required Claude SDK and MCP integration |
| `docs/` | experiment, XTC, evaluation, and workflow documentation |
| `environment.yml` | validated conda and pip dependency set |
| `psana_env.sh` | activation and configuration validation |

See [`docs/PSANA_XTC.md`](docs/PSANA_XTC.md) for data access and
[`docs/AUTOMASK.md`](docs/AUTOMASK.md) for the masking workflow.
