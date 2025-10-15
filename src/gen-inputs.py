from pathlib import Path

import h5py
import click
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import maskers, masking


def _load_stim_arrays(data_dir):
    """
    Parameters
    ----------
    data_dir : str
    """
    clip_feats = np.load(Path(data_dir, "clip-features", "features.npy"))
    clip_fnames = np.genfromtxt(
        Path(data_dir, "clip-features", "file_names.txt"), dtype=str
    )
    clip_fnames = [Path(f).stem for f in clip_fnames]

    return clip_feats, clip_fnames


def _load_brain_arrays(sub_name, roi, data_dir):
    """
    Parameters
    ----------
    sub_name : str
    roi : str
    data_dir : str
    """
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

    return y_vals, stim_names, session_labels


def _clean_inputs(stim_arr, y_arr, y_labels, x_arr, x_labels, cv_strategy="image"):
    """
    Clean provided brain, feature embedding arrays to
    (1) drop stimuli with less than three repetitions and
    (2) sort alphabetically by stimulus label s.t. reptitions
        are contiguous.

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
        Strategy for cross-validation. Must be in ["image", "category"].
        Note that "category" will return `incl_labels` corresponding
        to image categories (e.g., 'acorn')
        and "image" will return `incl_labels` corresponding
        to image identities (e.g., 'acorn_01b').

    Returns
    -------
    sorted_stim : np.arr
        Shape (n_unique, )
    y_matrix : np.arr
        Shape (3, n_unique, n_features)
    X_matrix : np.arr
        Shape (n_unique, dim_clip_embed)
    """
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

    # re-sort by stimulus label, after dropping stimuli with < 3 reps
    sort_idx = np.argsort(stim_arr[y_mask])
    sorted_stim = stim_arr[y_mask][sort_idx]

    x_unique, _ = np.unique(sorted_stim, return_inverse=True)
    assert np.all(x_unique == incl_labels)  # quick sanity check

    n_repeats = len(sorted_stim) // len(x_unique)
    assert n_repeats == 3  # another quick qa

    incl_y_arr = y_arr[y_mask][sort_idx]
    incl_y_labels = y_labels[y_mask][sort_idx]
    incl_x_arr = np.repeat(x_arr[x_mask], n_repeats, axis=0)

    # reshape according to cv_strategy
    if cv_strategy == "image":
        incl_labels = sorted_stim

    if cv_strategy == "category":
        incl_labels = np.asarray([stim.rsplit("_", 1)[0] for stim in sorted_stim])

    return incl_labels, incl_y_arr, incl_y_labels, incl_x_arr


def gen_inputs(sub_name, roi, cv_strategy, data_dir):
    """
    Parameters
    ----------
    sub_name : str
    roi : str
    cv_strategy : str
    data_dir : str
    """
    y_vals, stim_names, session_labels = _load_brain_arrays(sub_name, roi, data_dir)
    clip_feats, clip_fnames = _load_stim_arrays(data_dir)

    stim_vec, y_matrix, y_sessions, X_matrix = _clean_inputs(
        stim_names,
        y_vals,
        session_labels,
        clip_feats,
        clip_fnames,
        cv_strategy=cv_strategy,
    )

    return stim_vec, y_matrix, y_sessions, X_matrix


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option(
    "--cv_strategy",
    default=None,
    help="Strategy for cross-validation. Must be 'image' or 'category'",
)
@click.option(
    "--data_dir", default="/Users/emdupre/Desktop/things-encode", help="Data directory."
)
def main(sub_name, roi, cv_strategy, data_dir):
    """
    Create trialwise inputs for voxelwise encoding models on THINGS data using
    existing CLIP embeddings (previously generated using thingsvision).
    """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        err_msg = f"Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    cv_strategies = ["image", "category"]
    if cv_strategy not in cv_strategies:
        err_msg = f"Unrecognized cross-validation strategy {cv_strategy}"
        raise ValueError(err_msg)

    stim_vec, y_matrix, y_sessions, X_matrix = gen_inputs(
        sub_name, roi, cv_strategy, data_dir
    )

    out_stim = Path(
        data_dir, "encoding-inputs", f"{sub_name}_{cv_strategy}_outerCV_groups.txt"
    )
    if not out_stim.is_file():
        out_stim.parent.mkdir(exist_ok=True, parents=True)
        np.savetxt(out_stim, stim_vec, fmt="%s")

    if roi is not None:
        out_y_matrix = Path(
            data_dir, "encoding-inputs", f"{sub_name}_{roi}_brain_responses.npy"
        )
        if not out_y_matrix.is_file():
            out_y_matrix.parent.mkdir(exist_ok=True, parents=True)
            np.save(out_y_matrix, y_matrix)
    else:
        out_y_matrix = Path(
            data_dir, "encoding-inputs", f"{sub_name}_brain_responses.npy"
        )
        if not out_y_matrix.is_file():
            out_y_matrix.parent.mkdir(exist_ok=True, parents=True)
            np.save(out_y_matrix, y_matrix)

    out_y_sessions = Path(data_dir, "encoding-inputs", f"{sub_name}_innerCV_groups.txt")
    if not out_y_sessions.is_file():
        out_y_sessions.parent.mkdir(exist_ok=True, parents=True)
        np.savetxt(out_y_sessions, y_sessions, fmt="%s")

    out_X_matrix = Path(data_dir, "encoding-inputs", f"{sub_name}_stim_features.npy")
    if not out_X_matrix.is_file():
        out_X_matrix.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_X_matrix, X_matrix)


if __name__ == "__main__":
    main()
