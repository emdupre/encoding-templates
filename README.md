# encoding-templates

Experiments with encoding models in the THINGS dataset.


## Running on HPCs

Note that these analyses benefit from GPU availability;
as such, there is a containerized workflow tailed to 
[Alliance Canada](https://www.alliancecan.ca/)'s [Rorqual cluster](https://docs.alliancecan.ca/wiki/Rorqual/).

To re-generate the supporting Apptainer image:

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
salloc --account=<RRG-ACCOUNT> --gpus-per-node=h100_3g.40gb:1 --time=00:05:00 --mem=1G

nvidia-smi
apptainer test --nv things-apptainer.sif
```

the actual analyses can be re-launched using:

```
sbatch encoding.sbatch
```