#!/usr/bin/env bash
# Evaluate the selected production recipe. Pass eval.phase=validate only after
# parameters have been fixed on the fit runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "### production union (combo)"
python -m automask.studies.sweep_hyperparameters experiment=production "$@"
