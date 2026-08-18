# Reading raw XTC with psana

## First-time setup on a cluster (Sherlock)

Done once. Everything except the symlinks needs internet, so run it on the
**login node** — Sherlock's compute nodes have no direct outbound network.
Nothing here is CPU-heavy; it is download and I/O.

**Do not put any of this in `$HOME`.** Sherlock's home quota is small and a
psana env is several GB. Use group storage.

```bash
SW=/home/groups/darve/hippowal/sw          # conda + envs live here
mkdir -p "$SW" && cd "$SW"

# 1. conda + mamba, self-contained -- no dependence on cluster module names.
#    Miniforge3 ships both; Mambaforge was merged into it and is retired.
curl -L -o miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$SW/miniforge"
source "$SW/miniforge/etc/profile.d/conda.sh"
command -v mamba || conda install -y -n base -c conda-forge mamba

# 2. psana AND the project's other compiled deps, in ONE solve.
#    One solve on purpose: adding pyFAI or scikit-image afterwards lets the
#    solver bump numpy under psana, which breaks it in ways that surface as
#    segfaults rather than import errors. `mamba` here only to get a faster
#    solve and a readable conflict report -- it does not change the result.
mamba create -y -p "$SW/envs/ana-4.0.66-py311" -c lcls-i -c conda-forge \
    psana=4.0.66 python=3.11 \
    numpy h5py scipy scikit-image matplotlib pyfai hydra-core tifffile pytest

# 3. the package itself, without letting pip touch the compiled stack
conda activate "$SW/envs/ana-4.0.66-py311"
cd /home/users/hippowal/LCLS
pip install -e . --no-deps

# 4. agent runtime; pinned to the version validated with this psana solve
python -m pip install claude-agent-sdk==0.2.139

# 5. official SLAC detector adapters used by run profiling
mkdir -p "$SW/src"
git clone https://github.com/slac-lcls/smalldata_tools.git "$SW/src/smalldata_tools"
cd "$SW/src/smalldata_tools"
git checkout 5cf5c0ab7830f93bbc6213f7480b7a59322008bf
```

`mamba` and `conda` are interchangeable for `create`/`install` — same channels,
same flags, same resulting env. Prefer `mamba` for solving. Keep **`conda
activate`** for activation: `mamba activate` needs its own shell hook, whereas
the conda hook is the one `psana_env.sh` sources, so mixing them is the usual
way to end up with an env that "activated" without taking effect.

Then point the repo at the data and record the paths:

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
`PSANA_PSDM` only ever holds symlinks, so it needs no space — but it must be
writable. `PSANA_CONDA_SH` matters because a batch job starts without `conda`
on `PATH`.

Finally, verify from a compute node. Source the environment first so
`SIT_PSDM_DATA` resolves to group storage; the setup helper then builds the
layout psana requires and pulls one calibrated frame to prove it works:

```bash
source psana_env.sh
python -m automask.dev.setup_psdm_layout
# When a complete local run is present:
python -m automask.dev.setup_psdm_layout --check --run <RUN>
```

A mis-wired calib directory makes psana return `None` or uncalibrated data
**without raising**, so that stage checks an actual frame rather than the
existence of a directory. If it prints `det.calib() -> shape (2, 512, 1024)`,
the environment is good and the study stages can run.

At the time of writing, this machine's shared `xtc/` contains only one stream
each for runs 378, 389, and 396; run 475 is not present. Environment activation,
imports, and the PSDM/calibration link farm can still be verified, but the
frame-level check requires a locally available run that psana can decode.


Raw XTC is the production source for profiling and selected-shot reductions.

```bash
source psana_env.sh
# Optional: prewarm the ImageStore cache so evaluation stays numpy-only.
python -m automask.producers.build_images --run 389 475
```

`psana_env.sh` sets the `SIT_*` variables and activates the expected psana
environment. `automask.io.read_xtc.local_run_source()` describes the explicit
stream files and configures calibration lookup when opened.

Run 389 is incomplete locally: only truncated stream `s00` is available. It
cannot produce a complete bright-only sum or a verified run-specific reference
mask.
