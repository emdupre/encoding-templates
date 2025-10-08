from pathlib import Path
from collections import defaultdict

import click
import scipy
import sklearn
import numpy as np
from sklearn import pipeline
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from sklearn.preprocessing import StandardScaler
from himalaya.scoring import correlation_score, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_validate


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


def ridgeCV_model(
    X_matrix, y_matrix, groups=None, scoring=r2_score, inner_cv=True, inner_groups=None
):
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
    inner_cv : Bool
        Whether or not to perform nested cross-validation
    inner_groups : np.arr, Optional
        If performing nested cross-validation, group labels for inner_cv,
        corresponding to scanning session.
        Expected shape  of (n_samples, n_repeats)
    """
    # scores = defaultdict(list)

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    outer_cv = GroupKFold()
    alphas = np.logspace(1, 20, 20)
    estimator = RidgeCV(
        alphas=alphas,
        cv=outer_cv,
        solver_params=dict(
            n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
        ),
    )

    if inner_cv:
        inner_cv = GroupKFold()
        if inner_groups is None:
            raise ValueError("Must provide inner_groups to use inner_cv")

        for train, test in outer_cv.split(X_matrix, y_matrix, groups=groups):
            pass

    else:
        sklearn.set_config(enable_metadata_routing=True)
        scores = cross_validate(
            estimator,
            X_matrix,
            y=y_matrix,
            cv=outer_cv,
            # groups=groups,
            scoring=scoring,
            error_score="raise",
            params={"groups": groups},
        )
        # for train, test in outer_cv.split(X_matrix, y_matrix, groups=groups):
        #     # pl.set_params(ridgecv__cv_groups=groups[train])
        #     pl.fit(X_matrix[train], y_matrix[train])
        #     fold_score = pl.score(X_matrix[test], y_matrix[test])
        #     fold_alphas = pl[-1].best_alphas_
        #     scores["fold_score"].append(fold_score)
        #     scores["fold_alpha"].append(fold_alphas)
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
    "--inner_cv",
    is_flag=True,
    help="Whether or not to run nested cross-validation, "
    "cross-validating session number in inner CV.",
)
def main(sub_name, roi, cv_strategy, average, data_dir, inner_cv=True):
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

    # TODO: Remove this when tested
    if roi is not None:
        raise NotImplementedError

    groups = np.loadtxt(
        Path(data_dir, f"{sub_name}_{cv_strategy}_outerCV_groups.txt"), dtype=np.str_
    )
    inner_groups = np.loadtxt(
        Path(data_dir, f"{sub_name}_innerCV_groups.txt"), dtype=np.str_
    )
    X_matrix = np.load(Path(data_dir, f"{sub_name}_stim_features.npy"))
    if roi is not None:
        y_matrix = np.load(Path(data_dir, f"{sub_name}_{roi}_brain_responses.npy"))
    else:
        y_matrix = np.load(Path(data_dir, f"{sub_name}_brain_responses.npy"))

    if average:
        # cannot use inner CV when averaging
        inner_cv = False
        inner_groups = None

        # NOTE: shapes hard-coded for three repetitions, 4174 images, THINGS dataset
        groups = groups[::3]
        X_matrix = X_matrix[::3]
        y_matrix = np.mean(y_matrix.reshape(len(groups), 3, y_matrix.shape[-1]), axis=1)

    scores = ridgeCV_model(
        X_matrix,
        y_matrix,
        groups=groups,
        scoring=r2_score,
        inner_cv=inner_cv,
        inner_groups=inner_groups,
    )

    np.save(f"{sub_name}_cv-{cv_strategy}_r2_scores_clip.npy", scores.cpu())


if __name__ == "__main__":
    main()
