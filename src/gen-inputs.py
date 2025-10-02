from pathlib import Path

import h5py
import click
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import maskers, masking


def _subset_arrays(stim_arr, y_arr, y_labels, x_arr, x_labels, cv_strategy="image"):
    """
    Parameters
    ----------
    stim_arr : np.arr or list
        Shape (n_stim, )
    y_arr : np.arr or list
        Shape (n_stim, n_features)
    y_labels : np.arr or list
        Shape (n_stim,)
    x_arr : np.arr or list
        Shape (n_stim, dim_clip_embed)
    x_labels : np.arr or list
        Shape (n_stim,)
    cv_strategy : str
        Strategy for cross-validation. Must be in ["image", "session", "category"].
        Note that "session" will divide stimulus repeats s.t. across-session repeats
        are used for cross-validation, "category" will divide images such that images
        from the same semantic category are retained within cross-validation folds,
        and "image" will keep only individual image repetitions within folds,
        ignoring session effects and category information across folds.

    Returns
    -------
    sorted_stim : np.arr
        Shape (n_unique, )
    y_matrix : np.arr
        Shape (3, n_unique, n_features)
    X_matrix : np.arr
        Shape (n_unique, dim_clip_embed)
    """
    cv_strategies = ["image", "session", "category"]
    if cv_strategy not in cv_strategies:
        warn_msg = "Unrecognized cross-validation strategy {cv_strategy}"
        raise UserWarning(warn_msg)

    # type coercion
    stim_arr = np.asarray(stim_arr)
    y_arr = np.asarray(y_arr)
    y_labels = np.asarray(y_labels)
    x_arr = np.asarray(x_arr)

    label, counts = np.unique(stim_arr, return_counts=True)
    incl_labels = label[counts == 3]

    # filter clip features by incl_labels
    x_mask = [x_label in incl_labels for x_label in x_labels]

    # consider only stimuli with at least three repetitions
    y_mask = [s in incl_labels for s in stim_arr]
    n_unique = np.sum(y_mask)

    # re-sort by stimulus label, after dropping stimuli with < 3 reps
    sort_idx = np.argsort(stim_arr[y_mask])

    # reshape according to cv_strategy
    if cv_strategy == "image":
        rep_y_arr = np.reshape(
            y_arr[y_mask][sort_idx], (3, n_unique // 3, y_arr.shape[-1]), order="F"
        )
        rep_y_labels = np.reshape(
            y_labels[y_mask][sort_idx], (3, n_unique // 3), order="F"
        )
        x_arr = x_arr[x_mask]

    if cv_strategy == "session":
        ses_y_labels = np.reshape(y_labels[y_mask][sort_idx], (n_unique // 3, 3))
        ses_sort_idx = np.argsort(ses_y_labels)

        sort_y_labels = np.take_along_axis(ses_y_labels, ses_sort_idx, axis=1)
        rep_y_labels = np.swapaxes(sort_y_labels, 0, 1)

        ses_y_arr = np.reshape(
            y_arr[y_mask][sort_idx], (n_unique // 3, 3, y_arr.shape[-1])
        )
        sort_y_arr = np.take_along_axis(ses_y_arr, ses_sort_idx[..., None], axis=1)
        rep_y_arr = np.swapaxes(sort_y_arr, 0, 1)
        x_arr = x_arr[x_mask]

    if cv_strategy == "category":
        sorted_stim = stim_arr[y_mask][sort_idx]
        categ = np.asarray([stim.rsplit("_", 1)[0] for stim in sorted_stim])
        _, split_idx, counts = np.unique(categ, return_index=True, return_counts=True)

        rep_y_arr = np.split(y_arr[y_mask][sort_idx], split_idx)[
            1:
        ]  # dropping first empty split
        rep_y_labels = np.split(y_labels[y_mask][sort_idx], split_idx)[1:]

        x_unique, x_inv = np.unique(sorted_stim, return_inverse=True)
        assert np.all(x_unique == incl_labels)  # not best practice but a quick check

        x_arr = np.split(x_arr[x_mask][x_inv], split_idx)[1:]
        incl_labels = np.split(incl_labels[x_inv], split_idx)[1:]

    return incl_labels, rep_y_arr, rep_y_labels, x_arr


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option("--seed", default=0, help="Random seed.")
@click.option(
    "--cv_strategy",
    default=None,
    help="Strategy for cross-validation. Must be 'image', 'session', or 'category'",
)
@click.option(
    "--data_dir", default="/Users/emdupre/Desktop/things-encode", help="Data directory."
)
def main(sub_name, roi, seed, cv_strategy, data_dir):
    """
    Create trialwise inputs for voxelwise encoding models on THINGS data using
    existing CLIP embeddings (previously generated using thingsvision).
    """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        warn_msg = "Unrecognized ROI {roi}"
        raise UserWarning(warn_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        warn_msg = "Unrecognized subject {sub_name}"
        raise UserWarning(warn_msg)

    annot_fname = f"{sub_name}_task-things_desc-perTrial_annotation.tsv"
    beta_fname = f"{sub_name}_task-things_space-T1w_model-fitHrfGLMdenoiseRR_stats-trialBetas_desc-zscore_statseries.h5"

    beta_h5 = h5py.File(Path(data_dir, "betas", beta_fname), "r")

    if roi is not None:
        mask = nib.nifti1.Nifti1Image(
            np.array(beta_h5["mask_array"]), affine=np.array(beta_h5["mask_affine"])
        )

        roi_fname = (
            f"{sub_name}_task-floc_space-T1w*_roi-{roi}_*_desc-smooth_mask.nii.gz"
        )
        roi_nii = nib.load(
            next(Path(data_dir, "rois", sub_name).glob(roi_fname))
        )  # Shape (76, 90, 71)
        # plotting.view_img(roi_nii, bg_img=unmask_beta)
        # FFA ROI raises concern on visual inspection
        # (e.g., left FFA is two disconnected pieces of cortex).
        # Worth re-visiting processing steps.
        masker = maskers.NiftiMasker(mask_img=roi_nii).fit()

    clip_feats = np.load(Path(data_dir, "clip-features", "features.npy"))
    clip_fnames = np.genfromtxt(
        Path(data_dir, "clip-features", "file_names.txt"), dtype=str
    )
    clip_fnames = [Path(f).stem for f in clip_fnames]

    annot_df = pd.read_csv(Path(data_dir, "annot", annot_fname), sep="\t")
    annot_df = annot_df.loc[annot_df["exclude_session"] == False]
    annot_df = annot_df.loc[annot_df["atypical"] == False]

    # subset_idx = None
    y_vals = []
    stim_names = []
    session_labels = []

    for _, row in annot_df.iterrows():
        # this is ugly, but we know that sessions are always labeled ses-???
        # in the dataframe. since they're labelled as digits in the h5, munge them a bit
        ses_idx = str(int(row["session"][-3:]))
        run_idx = str(row["run"])

        # these are one-indexed rather than zero-indexed
        trial_idx = row["TrialNumber"] - 1

        try:
            if roi is not None:
                unmask_beta = masking.unmask(
                    beta_h5[ses_idx][run_idx]["betas"][trial_idx], mask
                )  # Shape (76, 90, 71)
                func_beta = masker.transform(unmask_beta)
            func_beta = beta_h5[ses_idx][run_idx]["betas"][trial_idx]

        # run-6, ses-08 of sub-06 is dropped from betas but not from data frame
        # https://github.com/courtois-neuromod/cneuromod-things/tree/main/THINGS
        # inelegant handling; ideally we'd scrub the data frame
        except KeyError:
            continue
        y_vals.append(func_beta.squeeze())

        stim_names.append(row["image_name"])
        session_labels.append(row["session"])

    stim_vec, y_matrix, y_sessions, X_matrix = _subset_arrays(
        stim_names,
        y_vals,
        session_labels,
        clip_feats,
        clip_fnames,
        cv_strategy=cv_strategy,
    )

    return stim_vec, y_matrix, y_sessions, X_matrix


if __name__ == "__main__":
    main()
