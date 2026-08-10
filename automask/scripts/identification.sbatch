#!/usr/bin/env bash
#SBATCH --job-name=identification
#SBATCH --output=automask/outputs/slurm/identification_%x_%j.out
#SBATCH --error=automask/outputs/slurm/identification_%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# The identification study, in the order the results depend on each other.
#
#   sbatch --time=00:20:00 --mem=8G automask/scripts/identification.sbatch setup 475
#   sbatch automask/scripts/identification.sbatch survey 475
#   sbatch automask/scripts/identification.sbatch gate   475
#   sbatch automask/scripts/identification.sbatch frames 475
#
# Add `-p <partition>` if your cluster has no usable default. The #SBATCH values
# below are sized for `frames`; override them at submit time for the cheap
# stages, as shown above.
#
# Nothing here runs on a login node -- Sherlock kills sustained CPU there, and
# psana lives in a conda env that the login shell may not have. Creating the
# `xtc`/`calib` symlinks is the only step that does NOT need a job:
#
#   cd /home/users/hippowal/LCLS
#   ln -sfn /home/groups/darve/hippowal/LCLS/xtc   xtc
#   ln -sfn /home/groups/darve/hippowal/LCLS/calib calib
#
# For an interactive session instead of batch:
#   srun --time=00:30:00 --mem=8G --pty bash
#   bash automask/scripts/identification.sbatch setup 475
#
# STAGES, and why they are separate jobs rather than one script.
#
#   setup   builds the $SIT_PSDM_DATA symlink farm psana needs (the repo's own
#           xtc/ and calib/ are NOT where psana looks), then proves it worked by
#           pulling one calibrated frame. A mis-wired calib directory makes
#           psana return uncalibrated data or None WITHOUT raising, so this
#           stage checks a frame rather than a directory. Run it once, and
#           re-run it if the symlinks or SIT_PSDM_DATA change.
#
#   survey  seconds, no detector frames. Lists which condition axes the run
#           actually has and how many groups each resolves into. If `delay` is
#           absent here, every later stage is working with a weaker axis and you
#           want to know that before spending an XTC pass.
#
#   gate    one XTC pass per condition axis (~800 frames each), then claim M2R:
#           does ANY axis move the sample signal enough to separate tau from S?
#           **If this comes back REFUTED, stop.** The two-stage estimator has no
#           identifying power on this run and `frames` cannot tell you anything
#           useful about it. This is the cheapest decisive measurement in the
#           study and it is deliberately its own job.
#
#   frames  claims A1, A3, M5, P3. Needs the frozen inputs (`automask/data/`),
#           the fold cache, and the condition cache `gate` already built. A1
#           rebuilds the fold cache on a miss, which is one more XTC pass.
#
# Everything writes to automask/outputs/ (gitignored). Reruns are cheap: the
# XTC passes are cached, so only the first job of each stage pays for them.
set -euo pipefail

STAGE="${1:?usage: identification.sbatch <setup|survey|gate|frames|all> <run>...}"
shift
case "${STAGE}" in
  setup|survey|gate|frames|all) ;;
  *) echo "unknown stage '${STAGE}' (setup|survey|gate|frames|all)" >&2; exit 2 ;;
esac
RUNS=("$@")
[[ ${#RUNS[@]} -gt 0 ]] || RUNS=(475)

cd "$(dirname "$0")/../.."
mkdir -p automask/outputs/slurm

# psana lives in a conda env, not in the module system; psana_env.sh sets the
# three SIT_* variables and activates it. Edit ENVP/PSDM there for Sherlock.
# shellcheck disable=SC1091
source psana_env.sh

echo "=== stage=${STAGE} runs=${RUNS[*]}  $(date -Is)"
python -c "import psana, automask; print('psana + automask import OK')"

case "${STAGE}" in
  setup)
    python -m automask.dev.setup_psdm_layout --check --run "${RUNS[0]}"
    ;;
  survey)
    python -m automask.identification.conditions "${RUNS[@]}"
    ;;
  gate)
    python -m automask.studies.loss_identification --claims M2R --runs "${RUNS[@]}"
    ;;
  frames)
    python -m automask.studies.loss_identification \
      --claims A1 A2 A3 M5 P3 --runs "${RUNS[@]}" --plots
    ;;
  all)
    python -m automask.identification.conditions "${RUNS[@]}"
    python -m automask.studies.loss_identification \
      --claims M2R A1 A2 A3 M5 P3 --runs "${RUNS[@]}" --plots
    ;;
esac

echo "=== done  $(date -Is)"
