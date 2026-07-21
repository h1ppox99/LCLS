#!/usr/bin/env bash
# Sweep the window-median detector over threshold K x TV weight W (with padding)
# on both eval runs. Supersedes studies/tv_window_median_sweep.py + kw_sweep.py.
set -euo pipefail
cd "$(dirname "$0")/.."
python studies/sweep_hyperparameters.py -m \
  stat=window_median regularization=tv mask_reg=pad combine=union \
  stat.params.k=1.5,2.0,2.5,3.0 \
  regularization.params.weight=5.0,10.0,15.0,20.0 \
  "$@"
