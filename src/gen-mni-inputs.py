from pathlib import Path

import ants
import h5py
import click
import joblib
import numpy as np
import nibabel as nib
from nilearn import masking, maskers
from templateflow import api as tflow


def warp_beta_imgs(
    y: np.ndarray,
    fixed: ants.core.ants_image.ANTsImage,
    xfm: str | Path,
    mask_img: nib.Nifti1Image,
    warped_mask: nib.Nifti1Image,
    interpolator: str = "bSpline",
) -> np.ndarray:
    unmask_y = masking.unmask(y, mask_img)
    moving = ants.from_nibabel_nifti(unmask_y)
    warped_ants = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=str(xfm),
        interpolator=interpolator,
    )
    warped_img = ants.to_nibabel_nifti(warped_ants)
    warped_y = masking.apply_mask(imgs=warped_img, mask_img=warped_mask)
    return warped_y


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option(
    "--data_dir",
    default="/home/emdupre/links/projects/def-pbellec/emdupre/things.betas",
    help="Data directory.",
)
def main(sub_name, data_dir):
    """
    Create MNI-space trialwise inputs for voxelwise encoding models on
    THINGS data using previously generated native-space inputs.
    """
    beta_fname = f"{sub_name}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stats-trialBetas_desc-zscore_statseries.h5"
    beta_h5 = h5py.File(Path(data_dir, "betas", beta_fname), "r")
    mask_img = nib.nifti1.Nifti1Image(
        np.array(beta_h5["mask_array"]), affine=np.array(beta_h5["mask_affine"])
    )
    # mask_nii = nib.load(Path(data_dir, "encoding-inputs", f"{sub_name}_brain_mask.nii.gz"))
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
        Path(data_dir, "encoding-inputs", f"{sub_name}_brain_responses.npy"),
        mmap_mode="r",
    )
    warped_mask = ants.to_nibabel_nifti(
        ants.apply_transforms(
            fixed=fixed,
            moving=ants.from_nibabel_nifti(mask_img),
            transformlist=str(xfm),
            interpolator="nearestNeighbor",
        )
    )

    warped_y_matrix = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(warp_beta_imgs)(
            y_matrix[n],
            xfm,
            mask_img,
            warped_mask,
        )
        for n in range(len(y_matrix))
    )

    np.save(
        Path(
            data_dir,
            "encoding-inputs",
            f"{sub_name}_MNI152NLin2009cAsym-brain_responses.npy",
        ),
        np.vstack(warped_y_matrix),
    )
