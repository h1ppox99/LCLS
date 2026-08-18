#!/usr/bin/env bash
# Sweep the black-hat detector over threshold K x TV weight W (with padding) on
# the fit runs. Supersedes the black-hat panel of studies/kw_sweep.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m automask.studies.sweep_hyperparameters -m \
  stat=blackhat regularization=tv mask_reg=pad \
  stat.params.k=1.5,2.0,2.5,3.0 \
  regularization.params.weight=1.0,2.0,5.0,10.0 \
  "$@"
