# Reading raw XTC with psana

Raw XTC is the production source for run profiling and selected-shot reductions.
For everyday use the environment is already built — just activate it.

## Using the data

```bash
source psana_env.sh
# Optional: prewarm the ImageStore cache so evaluation stays numpy-only.
python -m automask.producers.build_images --run 475
```

`psana_env.sh` sets the `SIT_*` variables and activates the psana environment.
Open a run with `automask.io.read_xtc.local_run_source()`, which names the
explicit stream files and wires up calibration lookup:

```python
from automask.io.read_xtc import local_run_source

source = local_run_source(475)
```

### Runs available locally

Runs are present in `/home/groups/darve/hippowal/LCLS/xtc` and `/home/groups/darve/hippowal/LCLS/calib`, which are symlinked into the repo. Make sure to symlink to your own group storage if you copy the repo. The following table shows the runs available in the local layout:

| run | streams | note |
| --- | --- | --- |
| 475 | s00–s04, complete | decodes fully; use this run |
| 389 | s03 only, truncated | open **by explicit path** — the run resolver rejects the incomplete layout |
| 378 | s00 only | partial |
| 396 | s00 only | partial |

For partial runs use `local_run_source(<run>)` (globs the present streams);
`exp=xppl1016922:run=<run>` fails on the incomplete layout. A truncated run
decodes cleanly up to its EOF, then warns without crashing.

---

## First-time environment build (admin, once)

Everything except the symlinks needs internet, so run it on the **login node** —
Sherlock's compute nodes have no outbound network. Nothing here is CPU-heavy.
**Do not put any of it in `$HOME`** (small quota); use group storage.

```bash
SW=/home/groups/darve/hippowal/sw          # conda + envs live here
mkdir -p "$SW" && cd "$SW"

# 1. conda + mamba, self-contained.
curl -L -o miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$SW/miniforge"
source "$SW/miniforge/etc/profile.d/conda.sh"
command -v mamba || conda install -y -n base -c conda-forge mamba

# 2. psana AND the compiled deps in ONE solve. Adding pyFAI/scikit-image after
#    the fact lets the solver bump numpy under psana and break it (segfaults, not
#    import errors), so pin them together.
mamba create -y -p "$SW/envs/ana-4.0.66-py311" -c lcls-i -c conda-forge \
    psana=4.0.66 python=3.11 \
    numpy h5py scipy scikit-image matplotlib pyfai pytest

# 3. the package, without letting pip touch the compiled stack
conda activate "$SW/envs/ana-4.0.66-py311"
cd /home/users/hippowal/LCLS
pip install -e . --no-deps

# 4. agent runtime, pinned to the validated version
python -m pip install claude-agent-sdk==0.2.139

# 5. official SLAC detector adapters used by run profiling
mkdir -p "$SW/src"
git clone https://github.com/slac-lcls/smalldata_tools.git "$SW/src/smalldata_tools"
cd "$SW/src/smalldata_tools"
git checkout 5cf5c0ab7830f93bbc6213f7480b7a59322008bf
```

`mamba` and `conda` are interchangeable for `create`/`install`; prefer `mamba`
for solving but keep `conda activate` for activation (it is the hook
`psana_env.sh` sources).

Then point the repo at the data and record the machine paths:

```bash
cd /home/users/hippowal/LCLS
ln -sfn /home/groups/darve/hippowal/LCLS/xtc   xtc
ln -sfn /home/groups/darve/hippowal/LCLS/calib calib

cat > psana_env.local <<'EOF'
PSANA_ENV=/home/groups/darve/hippowal/sw/envs/ana-4.0.66-py311
PSANA_PSDM=/home/groups/darve/hippowal/psdm
PSANA_CONDA_SH=/home/groups/darve/hippowal/sw/miniforge/etc/profile.d/conda.sh
SMALLDATA_TOOLS=/home/groups/darve/hippowal/sw/src/smalldata_tools
EOF
```

`psana_env.local` is gitignored, so machine paths never reach a commit.
`PSANA_PSDM` holds only symlinks (no space needed) but must be writable;
`PSANA_CONDA_SH` matters because a batch job starts without `conda` on `PATH`.

Finally, verify. A mis-wired calib directory makes psana return `None` or
uncalibrated data **without raising**, so this checks a real frame:

```bash
source psana_env.sh
python -m automask.research.dev.setup_psdm_layout
python -m automask.research.dev.setup_psdm_layout --check --run 475
```

If it prints `det.calib() -> shape (2, 512, 1024)`, the environment is good.
