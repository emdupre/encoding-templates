from pathlib import Path

import ants
import click
import numpy as np
import nibabel as nib
from nilearn import masking, maskers
from templateflow import api as tflow


def warp_beta_imgs(y):
    unmask_y = masking.unmask(y, mask_img)
    moving = ants.from_nibabel_nifti(unmask_y)
    warped_ants = ants.apply_transforms(
        fixed=fixed, moving=moving, transformlist=str(xfm)
    )
    warped_img = ants.to_nibabel_nifti(warped_ants)
    warped_y = masking.apply_mask(imgs=warped_img, mask_img=warped_mask)
    return warped_y


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option(
    "--data_dir",
    default="/home/emdupre/links/projects/rrg-pbellec/emdupre/things.betas",
    help="Data directory.",
)
def main(sub_name, data_dir):
    """
    Create MNI-space trialwise inputs for voxelwise encoding models on
    THINGS data using previously generated native-space inputs.
    """
    mask_img = nib.load(
        Path(data_dir, "encoding-inputs", f"{sub_name}_brain_mask.nii.gz")
    )
    tmpl = tflow.get(
        "MNI152NLin2009cAsym", resolution=2, desc=None, suffix="T1w", extension="nii.gz"
    )
    fixed = ants.image_read(str(tmpl))

    xfm = Path(
        data_dir,
        "anat.smriprep",
        sub_name,
        "anat",
        f"{sub_name}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    )

    y_matrix = np.load(
        Path(data_dir, "encoding-inputs", f"{sub_name}_brain_responses.npy")
    )
    warped_mask = ants.to_nibabel_nifti(
        ants.apply_transforms(
            fixed=fixed,
            moving=ants.from_nibabel_nifti(mask_img),
            transformlist=str(xfm),
            interpolator="nearestNeighbor",
        )
    )
    warped_y_matrix = [warp_beta_imgs(y) for y in y_matrix]
    np.vstack(warped_y_matrix)
