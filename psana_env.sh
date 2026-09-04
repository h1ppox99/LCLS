#!/usr/bin/env bash
# Activate the project runtime. Machine paths belong in psana_env.local.

_automask_setup() {
    local here envp backend data_root name
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck disable=SC1091
    [ -f "$here/psana_env.local" ] && source "$here/psana_env.local"

    envp="${PSANA_ENV:-}"
    if ! python -c "import psana" >/dev/null 2>&1; then
        if [ -z "$envp" ]; then
            echo "psana_env.sh: psana is not active; set PSANA_ENV in" >&2
            echo "              $here/psana_env.local or source an LCLS psana env" >&2
            return 1
        fi
        [ -d "$envp" ] || {
            echo "psana_env.sh: no such conda env: $envp" >&2
            return 1
        }
        if [ -n "${PSANA_CONDA_SH:-}" ] && [ -f "$PSANA_CONDA_SH" ]; then
            # shellcheck disable=SC1090
            source "$PSANA_CONDA_SH"
        elif command -v conda >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            source "$(conda info --base)/etc/profile.d/conda.sh"
        else
            echo "psana_env.sh: conda is unavailable; set PSANA_CONDA_SH" >&2
            return 1
        fi
        conda activate "$envp" || return 1
    fi

    backend="${AUTOMASK_BACKEND:-auto}"
    case "$backend" in
        local)
            for name in AUTOMASK_XTC_DIR AUTOMASK_CALIB_DIR; do
                if [ -z "${!name:-}" ]; then
                    echo "psana_env.sh: set $name for the local backend" >&2
                    return 1
                fi
            done
            [ -d "$AUTOMASK_XTC_DIR" ] || {
                echo "psana_env.sh: XTC directory not found: $AUTOMASK_XTC_DIR" >&2
                return 1
            }
            [ -d "$AUTOMASK_CALIB_DIR" ] || {
                echo "psana_env.sh: calibration directory not found: $AUTOMASK_CALIB_DIR" >&2
                return 1
            }
            data_root="${AUTOMASK_DATA_ROOT:-$(dirname "$AUTOMASK_CALIB_DIR")}"
            export SIT_ROOT="${SIT_ROOT:-$data_root}"
            export SIT_PSDM_DATA="${SIT_PSDM_DATA:-$data_root}"
            export AUTOMASK_XTC_DIR AUTOMASK_CALIB_DIR
            ;;
        slac)
            for name in SIT_PSDM_DATA SIT_ROOT SIT_DATA; do
                if [ -z "${!name:-}" ]; then
                    echo "psana_env.sh: $name is unset for the SLAC backend" >&2
                    echo "              source the standard LCLS psana setup first" >&2
                    return 1
                fi
            done
            ;;
        auto) ;;
        *) echo "psana_env.sh: invalid AUTOMASK_BACKEND=$backend" >&2; return 1 ;;
    esac
    export AUTOMASK_BACKEND="$backend"
    if [ -n "${AUTOMASK_CACHE_DIR:-}" ]; then
        export AUTOMASK_CACHE_DIR
    fi

    if [ -n "${SMALLDATA_TOOLS:-}" ]; then
        [ -d "$SMALLDATA_TOOLS/smalldata_tools" ] || {
            echo "psana_env.sh: invalid SMALLDATA_TOOLS=$SMALLDATA_TOOLS" >&2
            return 1
        }
        export PYTHONPATH="$SMALLDATA_TOOLS${PYTHONPATH:+:$PYTHONPATH}"
        export SMALLDATA_TOOLS
    fi

    python -c "import psana, smalldata_tools, claude_agent_sdk, mcp, automask" \
        >/dev/null 2>&1 || {
        echo "psana_env.sh: required runtime imports failed; run lcls-agent doctor" >&2
        return 1
    }
    echo "automask ready: backend=$AUTOMASK_BACKEND python=$(command -v python)"
}

if _automask_setup; then
    :
else
    (return 0 2>/dev/null) && return 1 || exit 1
fi
