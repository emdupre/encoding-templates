#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --gpus-per-node=1
#SBATCH --mem=15G
#SBATCH --time=0-01:00

module load apptainer/1.3.4
module load cuda

nvidia-smi

apptainer exec --nv things-encode.sif \
    bash -c 'source /.venv/bin/activate ; uv run python things-encode/code/run-encoding-models.py'