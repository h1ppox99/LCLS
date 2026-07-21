#!/usr/bin/env bash
# Sweep the variance detector over threshold K x TV weight W on both eval runs.
# Supersedes studies/tv_variance_sweep.py. Results -> outputs/sweeps/<ts>/results.csv.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m automask.studies.sweep_hyperparameters -m \
  stat=variance regularization=tv mask_reg=none combine=union \
  stat.params.k=2.0,2.5,3.0,3.5 \
  regularization.params.weight=5.0,10.0,15.0,20.0 \
  "$@"
