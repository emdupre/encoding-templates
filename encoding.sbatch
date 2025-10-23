#!/bin/bash
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3
#SBATCH --time=00:25:00
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=things-encode

module load apptainer/1.3.4
module load cuda
nvidia-smi

PROJECT=$HOME/links/projects/rrg-pbellec/emdupre/things.betas
subjects=("sub-01" "sub-02" "sub-03" "sub-06")
cv_strategy='kfold'
engine='himalaya'

echo "Now running : "
echo apptainer exec --nv $PROJECT/things-apptainer.sif \
    bash -c "source /.venv/bin/activate ; "\
    "uv run python ${PROJECT}/encoding-templates/src/encoding.py "\
    "--sub_name=${subjects[$SLURM_ARRAY_TASK_ID]}" \
    "--data_dir=${PROJECT}" \
    "--average" \
    "--cv_strategy=${cv_strategy}" \
    "--engine=${engine}"
echo " "

apptainer exec --nv $PROJECT/things-apptainer.sif \
    bash -c "source /.venv/bin/activate ; \
    uv run python ${PROJECT}/encoding-templates/src/encoding.py \
    --sub_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=${PROJECT} \
    --average \
    --cv_strategy=${cv_strategy} \
    --engine=${engine}"


# echo ${subjects[$SLURM_ARRAY_TASK_ID]}
# source $PROJECT/things.env/bin/activate
# python $PROJECT/encoding-templates/src/encoding.py \
#     --sub_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
#     --data_dir=$PROJECT \
#     --average \
#     --cv_strategy=${cv_strategy} \
#     --engine=${engine}