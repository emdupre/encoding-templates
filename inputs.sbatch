#!/bin/bash
#SBATCH --job-name=things-betas
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --time=00:10:00
#SBATCH --account=rrg-pbellec

PROJECT=$HOME/links/projects/rrg-pbellec/emdupre/things.betas
subjects=("sub-01" "sub-02" "sub-03" "sub-06")

source $PROJECT/things.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/encoding-templates/src/gen-inputs.py \
    --sub_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT
