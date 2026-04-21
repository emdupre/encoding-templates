#!/bin/bash
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0
#SBATCH --time=00:25:00
#SBATCH --account=def-pbellec
#SBATCH --job-name=nsd-embeddings

module load cuda
nvidia-smi

PROJECT=$HOME/links/projects/def-pbellec/emdupre/nsd-algonauts/

subjects=("subj01" "subj02" "subj03" "subj04" "subj05" "subj06" "subj07" "subj08")
subj_name=${subjects[$SLURM_ARRAY_TASK_ID]}

file_path=train/${PROJECT}/${subj_name}/training_split/training_images

source $PROJECT/thingsvision/bin/activate

echo "Now running : "
echo "uv run python ${PROJECT}/encoding-templates/src/gen-embeddings.py "\
     "--input_dir=${file_path}" \
     "--output_dir=${file_path}/clip_embed "
echo " "

uv run python ${PROJECT}/encoding-templates/src/gen-embeddings.py \
    --input_dir=${file_path} \
    --output_dir=${file_path}/clip_embed
