# encoding-templates

Code for encoding models of the THINGS dataset, primarily focussing on visual features.

To re-generate the accompanying Apptainer image:

```
module load apptainer/1.3.4
apptainer cache list
apptainer cache clean
apptainer build --notest things-apptainer.sif apptainer.def
```

then, we can confirm it can access the GPU in a Rorqual HPC interactive job with:

```
module load apptainer/1.3.4
module load cuda
salloc --account=<RRG-ACCOUNT> --gpus-per-node=h100_3g.40gb:1 --time=00:10:00 --mem=10G

nvidia-smi
apptainer test --nv things-apptainer.sif
```

the actual analyses can be re-launched using:

```
sbatch src/encoding.sbatch
```