#!/usr/bin/bash
#SBATCH -N 6
#SBATCH --mem=100G
#SBATCH -t 0-1:00 # time (D-HH:MM)

source ~/.bashrc
export PYTHONUNBUFFERED=1
python3 src/model_apply.py "$@"
set -eu
