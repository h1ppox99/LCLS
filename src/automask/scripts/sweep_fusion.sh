#!/usr/bin/env bash
# Sweep the continuous-fusion combiners on the full production 3-detector stack.
# Supersedes studies/fusion_stats.py (weighted-sum + Mahalanobis vs the union combo).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### weighted-sum fusion: threshold K"
python studies/sweep_hyperparameters.py -m \
  experiment=production combine=weighted_sum \
  combine.params.k=2.5,3.0,3.5,4.0 \
  "$@"

echo "### mahalanobis fusion: threshold K"
python studies/sweep_hyperparameters.py -m \
  experiment=production combine=mahalanobis \
  combine.params.k=2.0,3.0,4.0,5.0 \
  "$@"
