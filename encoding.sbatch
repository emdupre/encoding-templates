#!/bin/bash
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3
#SBATCH --time=00:10:00
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=things-encode
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=elizabethm.dupre@gmail.com

module load cuda
nvidia-smi

PROJECT=$HOME/links/projects/rrg-pbellec/emdupre/things.betas
subjects=("sub-01" "sub-02" "sub-03" "sub-06")

source $PROJECT/things.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/encoding-templates/src/encoding.py \
    --sub_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT \
    --cv_strategy="kfold" \
    --engine="himalaya" \
    # --average