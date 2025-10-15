# encoding-templates

Code for encoding models of the THINGS dataset, primarily focussing on visual features.

To re-generate the accompanying Apptainer image:

```
module load apptainer/1.3.4
apptainer cache list
apptainer cache clean
apptainer build --notest things-encode.sif things-encode-sif.def
```

then, we can confirm it can access the GPU in a Beluga HPC interactive job with:

```
module load apptainer/1.3.4
module load cuda
salloc --ntasks=1 --gpus-per-task=1 --account=<RRG-GROUP> --time=0:10:0 --mem=500M

nvidia-smi
apptainer test --nv things-encode.sif
```

the actual analyses can be re-launched using:

```
sbatch things-encode/slurm-submission.sh
```