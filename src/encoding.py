from pathlib import Path

import click
import scipy
import numpy as np
from sklearn import pipeline
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from himalaya.scoring import correlation_score, r2_score


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
    X_matrix, y_matrix, groups=None, scoring=r2_score, inner_CV=True, inner_groups=None
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
    # Reshape to n_samples, n_features, n_repeats
    # y = np.moveaxis(y_matrix, [0, 2], [2, 1])

    # Implement standard scaler
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix, y_matrix)

    alphas = np.logspace(1, 20, 20)
    outer_cv = GroupKFold()
    inner_cv = GroupKFold()

    estimator = RidgeCV(
        alphas=alphas,
        cv=outer_cv,
        solver_params=dict(
            n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
        ),
    )

    for train, test in outer_cv.split(X_matrix, y_matrix, groups):
        if inner_cv:
            if inner_groups is None:
                raise ValueError("Must provide inner_groups to use inner_cv")

            # flatten data to support inner_cv splits
            n_repeats = y_matrix.shape[-1]
            X_train = np.repeat(X_matrix[train], n_repeats, axis=0)
            y_train = np.hstack(y_matrix[train]).T

            # Split inner cross-validation over sessions
            inner_splits = inner_cv.split(
                inner_groups[train].ravel(),
                groups=np.unique(inner_groups[train], return_inverse=True)[-1],
            )

            # Update estimator with inner cross-validator
            estimator.set_params(cv=inner_splits)

        else:
            X_train = X_matrix[train]
            y_train = y_matrix[train]

        # Fit the regression model on training data
        estimator.fit(X_train, y_train)

        # Test scores
        test_prediction = estimator.predict(X_matrix[test])
        test_score = scoring(y_matrix[test], test_prediction)

        best_alphas = estimator.best_alphas_

    return scores, best_alphas


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option("--cv_strategy", default="image", help="Cross-validation strategy")
@click.option(
    "--average", is_flag=True, help="Average repeat image presentations across folds."
)
@click.option(
    "--data_dir",
    default="/home/emdupre/projects/rrg-pbellec/emdupre/things-encode",
    help="Data directory.",
)
def main(sub_name, roi, cv_strategy, average, data_dir):
    """ """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        warn_msg = "Unrecognized ROI {roi}"
        raise UserWarning(warn_msg)

    if roi is not None:
        raise NotImplementedError

    sub_names = ["sub-01", "sub-02", "sub-03"]
    if sub_name not in sub_names:
        warn_msg = "Unrecognized subject {sub_name}"
        raise UserWarning(warn_msg)

    mask = Path(
        data_dir,
        "masks",
        f"{sub_name}_task-things_space-T1w_label-brain_desc-unionNonNaN_mask.nii.gz",
    )

    ev = explainable_variance(y_test)

    np.save(f"{sub_name}_cv-{cv_strategy}_explained_var.npy", ev)
    np.save(f"{sub_name}_cv-{cv_strategy}_r2_scores_clip.npy", scores.cpu())
    np.save(f"{sub_name}_cv-{cv_strategy}_best_alphas_clip.npy", best_alphas.cpu())


if __name__ == "__main__":
    main()
