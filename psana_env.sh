#!/usr/bin/env bash
# psana_env.sh -- set up the environment to read the local xppl1016922 XTC.
#
#   source psana_env.sh
#
# Sets the three SIT_* variables psana needs and activates the psana conda env.
# Adjust ENVP / PSDM if you installed elsewhere.

# --- edit these two if your paths differ ----------------------------------
ENVP=/Data/hippolyte.wallaert/envs/ana-4.0.62          # the psana conda env
PSDM=/Data/hippolyte.wallaert/psdm                     # SIT_PSDM_DATA root
# --------------------------------------------------------------------------

export SIT_PSDM_DATA="$PSDM"
export SIT_ROOT="$PSDM/sit_root"        # any valid dir; global calib repo (unused here)
export SIT_DATA="$PSDM/data"            # must contain ExpNameDb/experiment-db.dat
mkdir -p "$SIT_ROOT/detector/calib/constants" "$SIT_DATA/ExpNameDb"

# minimal experiment registry: "<expnum> <instrument> <experiment>"
if [ ! -s "$SIT_DATA/ExpNameDb/experiment-db.dat" ]; then
    echo "1016922 xpp xppl1016922" > "$SIT_DATA/ExpNameDb/experiment-db.dat"
fi

# activate the psana env (works whether or not conda is already initialised)
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENVP"
else
    echo "conda not found on PATH; activate '$ENVP' manually" >&2
fi

echo "psana env ready:"
echo "  SIT_PSDM_DATA=$SIT_PSDM_DATA"
echo "  SIT_ROOT=$SIT_ROOT"
echo "  SIT_DATA=$SIT_DATA"
echo "  python: $(command -v python)"
