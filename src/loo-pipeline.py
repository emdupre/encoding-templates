from pathlib import Path

import click
import numpy as np
from nilearn.maskers import NiftiMasker
from fmralign import GroupAlignment, PairwiseAlignment
from fmralign.embeddings.parcellation import get_labels


def alignment(mask_img):
    """ """
    masker = NiftiMasker(mask_img=mask_img).fit()
    # Use only the first image to speed up the computation of the labels
    labels = get_labels(imgs[0], n_pieces=150, masker=masker)

    # We create a dictionary with the subject names as keys and the subjects data as values
    dict_alignment = dict(zip(subjects, masked_imgs))

    # We use Procrustes/scaled orthogonal alignment method
    template_estim = GroupAlignment(method="procrustes", labels=labels)
    template_estim.fit(X=dict_alignment, y="template")
    fitted_template = template_estim.template
    return fitted_template


def predict(fitted_template, masker):
    """ """
    left_out_data = masker.transform(left_out_subject)
    labels = get_labels(left_out_data, n_pieces=150, masker=masker)
    pairwise_estim = PairwiseAlignment(method="procrustes", labels=labels).fit(
        left_out_data, fitted_template
    )
    predictions_from_template = pairwise_estim.transform(left_out_data)
    pass


@click.command()
@click.option("--roi", default=None, help="Region-of-interest")
@click.option("--cv_strategy", default="image", help="Cross-validation strategy")
@click.option(
    "--average",
    is_flag=True,
    help="Average repeat image presentations before encoding. "
    "Note that this is incompatible with the 'image' cv_strategy",
)
@click.option(
    "--data_dir",
    default="/home/emdupre/projects/rrg-pbellec/emdupre/things-encode",
    help="Data directory.",
)
@click.option(
    "--engine",
    default="himalaya",
    help="Engine for running encoding analyses. Must be either 'sklearn' "
    "or 'himalaya'. Note only the latter is GPU compatiable.",
)
def main(roi, cv_strategy, average, data_dir, engine):
    """ """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    cv_strategies = ["image", "category"]
    if cv_strategy not in cv_strategies:
        err_msg = f"Unrecognized cross-validation strategy {cv_strategy}"
        raise ValueError(err_msg)

    if average and (cv_strategy == "image"):
        err_msg = "Cross-validation strategy 'image' is not compatible with 'average'"
        raise ValueError(err_msg)

    # TODO: Remove this when tested
    if roi is not None:
        raise NotImplementedError

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    groups = np.loadtxt(
        Path(data_dir, f"{sub_name}_{cv_strategy}_outerCV_groups.txt"), dtype=np.str_
    )
    ####################################
    # FIXME
    inner_groups = np.loadtxt(
        Path(data_dir, f"{sub_name}_innerCV_groups.txt"), dtype=np.str_
    )
    ####################################
    X_matrix = np.load(Path(data_dir, f"{sub_name}_stim_features.npy"))
    if roi is not None:
        y_matrix = np.load(Path(data_dir, f"{sub_name}_{roi}_brain_responses.npy"))
    else:
        y_matrix = np.load(Path(data_dir, f"{sub_name}_brain_responses.npy"))

    pass


if __name__ == "__main__":
    main()
