# xppl1016922 — local copy + automatic detector masking

Recovered local copy of LCLS experiment **`xppl1016922`** (XPP) and its
Jungfrau1M masking project. The verified notebook reference is available only
for run 475; run 389 raw XTC is incomplete.

## Layout

```
automask/          the masking project — an installable Python package;
                   see automask/README.md for its layout and API
lcls_agent/        the stdio MCP server the agent drives automask through
docs/              experiment + psana background and guides
                   (DATA_OVERVIEW, PSANA_XTC, DATA, AUTOMASK, EVALUATION, CONSISTENCY)
psana_env.sh       activate the ana-4.0.66-py311 conda env (for XTC / psana)
calib/ xtc/        the production data mirror (gitignored, symlinked — large)
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
environment (for example, `conda install -p "$ENVP" -c conda-forge pyfai`).

### Everyday setup

```bash
source psana_env.sh
```

For workflows that do not read XTC data or import psana, a normal Python
environment can instead use `python -m pip install -e .`.

## Using it

`automask` is a Python library — see [`automask/README.md`](automask/README.md)
for the package layout and API, and [`docs/AUTOMASK.md`](docs/AUTOMASK.md) for the
scripting workflow. The agent reaches the same operations through the stdio MCP
server in [`lcls_agent/`](lcls_agent/README.md); the checked-in `.mcp.json`
registers it for Claude Code — activate `psana_env.sh` first and approve the
project server on first use.

[`docs/`](docs/) covers the local data and psana/XTC access.
