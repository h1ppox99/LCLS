#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 slac SOFTWARE_ROOT" >&2
    echo "       $0 local SOFTWARE_ROOT XTC_DIR CALIB_DIR [CACHE_DIR]" >&2
    exit 2
}

[ "$#" -ge 2 ] || usage
mode="$1"
software_root="$(realpath -m "$2")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$mode" in
    slac) [ "$#" -eq 2 ] || usage ;;
    local) [ "$#" -ge 4 ] && [ "$#" -le 5 ] || usage ;;
    *) usage ;;
esac

if [ "$mode" = slac ]; then
    for name in SIT_PSDM_DATA SIT_ROOT SIT_DATA; do
        if [ -z "${!name:-}" ]; then
            echo "$name is unset. Source the standard LCLS psana setup first." >&2
            exit 1
        fi
    done
fi

if command -v mamba >/dev/null 2>&1; then
    solver="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    solver="$(command -v conda)"
else
    echo "Install Miniforge first so mamba or conda is on PATH." >&2
    exit 1
fi

mkdir -p "$software_root/envs" "$software_root/src"
env_prefix="$software_root/envs/automask-py311"
if [ -d "$env_prefix/conda-meta" ]; then
    "$solver" env update --prefix "$env_prefix" --file "$repo_root/environment.yml" --prune
else
    "$solver" env create --prefix "$env_prefix" --file "$repo_root/environment.yml"
fi

conda_base="$(conda info --base)"
conda_sh="$conda_base/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$conda_sh"
conda activate "$env_prefix"

smalldata="$software_root/src/smalldata_tools"
smalldata_commit="5cf5c0ab7830f93bbc6213f7480b7a59322008bf"
if [ ! -d "$smalldata/.git" ]; then
    git clone https://github.com/slac-lcls/smalldata_tools.git "$smalldata"
fi
git -C "$smalldata" fetch origin "$smalldata_commit"
git -C "$smalldata" checkout --detach "$smalldata_commit"
python -m pip install --no-deps -e "$repo_root"

config="$repo_root/psana_env.local"
if [ -e "$config" ] && [ "${AUTOMASK_OVERWRITE_CONFIG:-0}" != 1 ]; then
    echo "$config already exists; environment built but configuration not replaced." >&2
    echo "Set AUTOMASK_OVERWRITE_CONFIG=1 to replace it." >&2
    exit 1
fi
{
    printf 'PSANA_ENV=%q\n' "$env_prefix"
    printf 'PSANA_CONDA_SH=%q\n' "$conda_sh"
    printf 'SMALLDATA_TOOLS=%q\n' "$smalldata"
    printf 'AUTOMASK_BACKEND=%q\n' "$mode"
    if [ "$mode" = local ]; then
        printf 'AUTOMASK_XTC_DIR=%q\n' "$(realpath -m "$3")"
        printf 'AUTOMASK_CALIB_DIR=%q\n' "$(realpath -m "$4")"
        if [ "$#" -eq 5 ]; then
            printf 'AUTOMASK_CACHE_DIR=%q\n' "$(realpath -m "$5")"
        fi
    fi
} > "$config"

echo "Environment ready. Run: source psana_env.sh"
