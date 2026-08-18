#!/usr/bin/env bash

#SBATCH --job-name=automask-prod
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/users/hippowal/LCLS/slurm-%j.out
#SBATCH --error=/home/users/hippowal/LCLS/slurm-%j.err

set -euo pipefail

cd /home/users/hippowal/LCLS

source psana_env.sh
export MPLBACKEND=Agg
python -u automask/notebooks/production.py
