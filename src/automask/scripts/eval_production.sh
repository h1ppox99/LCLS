#!/usr/bin/env bash
# Evaluate the live production recipe across all eval runs, both combiners.
# The regression anchor (matches masking.main() on run 475).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "### production union (combo)"
python studies/sweep_hyperparameters.py experiment=production combine=union "$@"
echo "### production weighted-sum (combo_sum)"
python studies/sweep_hyperparameters.py experiment=production combine=weighted_sum "$@"
