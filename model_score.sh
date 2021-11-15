#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -t 0-1:00 # time (D-HH:MM)

source ~/.bashrc
export PYTHONUNBUFFERED=1
python3 model_score.py "$@"
set -eu
