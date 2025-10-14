from pathlib import Path

import click
import scipy
import sklearn
import numpy as np
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, LeaveOneOut


backend = set_backend("torch_cuda", on_error="warn")


def explainable_variance(data, bias_correction=True, do_zscore=True):
    """
    Compute explainable variance for a set of voxels.

    Parameters
    ----------
    data : array of shape (n_repeats, n_times, n_voxels)
        fMRI responses of the repeated test set.
    bias_correction: bool
        Perform bias correction based on the number of repetitions.
    do_zscore: bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.

    Returns
    -------
    ev : array of shape (n_voxels, )
        Explainable variance per voxel.
    """
    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    ev = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        ev = ev - (1 - ev) / (n_repeats - 1)
    return ev


def ridgeCV_model(X_matrix, y_matrix, groups=None, scoring_metric=r2_score):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image category.
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    """
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    outer_cv = GroupKFold()
    alphas = np.logspace(1, 20, 20)
    estimator = RidgeCV(
        alphas=alphas,
        cv=LeaveOneOut,
        solver_params=dict(
            n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
        ),
    )
    scorer = make_scorer(scoring_metric)
    sklearn.set_config(enable_metadata_routing=True)

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        error_score="raise",
        params={"groups": groups},
    )
    return scores


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
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
    "--scoring_metric",
    default="r2_score",
    # help="Whether or not to run nested cross-validation, "
    # "cross-validating session number in inner CV.",
)
def main(sub_name, roi, cv_strategy, average, data_dir, scoring_metric):
    """ """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = "Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        err_msg = "Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    cv_strategies = ["image", "category"]
    if cv_strategy not in cv_strategies:
        err_msg = "Unrecognized cross-validation strategy {cv_strategy}"
        raise ValueError(err_msg)

    if average and (cv_strategy == "image"):
        err_msg = "Cross-validation strategy 'image' is not compatible with 'average'"
        raise ValueError(err_msg)

    scoring_metrics = ["r2_score", "correlation_score"]
    if scoring_metric not in scoring_metrics:
        err_msg = "Unrecognized scoring metric {scoring_metric}"
        raise ValueError(err_msg)

    if scoring_metric == "r2_score":
        scoring = r2_score
    if scoring_metric == "correlation_score":
        scoring = correlation_score

    # TODO: Remove this when tested
    if roi is not None:
        raise NotImplementedError

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

    if average:
        # NOTE: shapes hard-coded for three repetitions, 4174 images, THINGS dataset
        groups = groups[::3]
        X_matrix = X_matrix[::3]
        y_matrix = np.mean(y_matrix.reshape(len(groups), 3, y_matrix.shape[-1]), axis=1)

    scores = ridgeCV_model(X_matrix, y_matrix, groups=groups, scoring=scoring)

    np.save(f"{sub_name}_cv-{cv_strategy}_r2_scores_clip.npy", scores.cpu())


if __name__ == "__main__":
    main()
