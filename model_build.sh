#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -t 0-1:00 # time (D-HH:MM)

source ~/.bashrc
export PYTHONUNBUFFERED=1
python3 src/model_build.py "$1"
set -eu

