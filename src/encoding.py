import os
from pathlib import Path
from collections import defaultdict

import click
import scipy
import cortex
import sklearn
import numpy as np
from cortex import db
from nilearn import masking
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from himalaya.scoring import correlation_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_validate, KFold, GroupKFold


os.environ["PATH"] += ":/Applications/Inkscape.app/Contents/MacOS/"


def plot_flatmap(
    best_scores, sub_name, mask_img, cv_strategy, scoring_metric="r2_score"
):
    """
    Parameters
    ----------
    nii : nib.Nifti
        voxel-wise data to project to flatmap
    sub_name : str
    """
    lh, rh = cortex.get_hemi_masks(subject=sub_name, xfmname="align_auto")
    db.save_mask(sub_name, "align_auto", "lh", lh)

    nii = masking.unmask(best_scores, mask_img)

    # https://gallantlab.org/pycortex/auto_examples/datasets/plot_vertex.html
    nii_vol = cortex.Volume(
        data=nii.get_fdata().T[rh],
        subject=sub_name,
        xfmname="align_auto",
        # vmin=(np.min(ffx_nii.dataobj) * 0.95),
        # vmax=(np.max(ffx_nii.dataobj) * 0.95),
        # cmap="magma",
    )

    out_name = f"{sub_name}_{cv_strategy}_encoding_{scoring_metric}_flatmap.png"
    fig = cortex.quickshow(nii_vol, sampler="nearest")
    cortex.quickflat.make_png(
        out_name,
        nii_vol,
        sampler="trilinear",
        curv_brightness=1.0,
        with_colorbar=True,
        colorbar_location="left",
        with_curvature=True,
        with_labels=False,
        with_rois=True,
        dpi=300,
        height=2048,
    )
    return fig


def plot_alphas_diagnostic(best_alphas, alphas, cv_fold=None, ax=None):
    """
    Adapted from gallantlab/himalaya
    BSD 3-Clause License
    Copyright (c) 2020, the himalaya developers All rights reserved.

    Plot a diagnostic plot for the selected alphas during cross-validation.

    To figure out whether to increase the range of alphas.

    Parameters
    ----------
    best_alphas : array of shape (n_targets, )
        Alphas selected during cross-validation for each target.
    alphas : array of shape (n_alphas)
        Alphas used while fitting the model.
    cv_fold : int or None
        Outer cross-validation fold, for labelling
    ax : None or figure axis

    Returns
    -------
    ax : figure axis
    """
    alphas = np.sort(alphas)
    n_alphas = len(alphas)
    indices = np.argmin(np.abs(best_alphas[None] - alphas[:, None]), 0)
    hist = np.bincount(indices, minlength=n_alphas)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    log10alphas = np.log(alphas) / np.log(10)
    ax.plot(log10alphas, hist, ".-", markersize=12, label=f"Outer-CV fold {cv_fold}")
    ax.set_ylabel("Number of targets")
    ax.set_xlabel("log10(alpha)")
    if cv_fold is not None:
        ax.legend()
    ax.grid("on")
    return ax


def plot_voxel_hist(
    sub_name, expl_var, best_scores, scoring_metric="r2_score", ax=None
):
    r"""
    Parameters
    ----------
    sub_name : str
        Subject name
    expl_var : np.arr
        Explainable variance, as calculated using
        .. math::
            \\frac{1}{N}\\sum_{i=1}^N\\text{Var}(y_i) - \\frac{N}{N-1}\\sum_{i=1}^N\\text{Var}(r_i)
    scores : np.arr
        Scores from the encoding model calculated using the scoring metric
    scoring_metric : str
        Scoring metric used in encoding model scoring, must be 'r2_score' or 'correlation_score'

    Returns
    -------
    ax : figure axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(
        expl_var,
        bins=np.linspace(0, 1, 100),
        log=True,
        histtype="step",
        label="Explainable variance",
    )
    ax.hist(
        best_scores,
        bins=np.linspace(0, 1, 100),
        log=True,
        histtype="step",
        label=(
            "$R^2$ values" if (scoring_metric == "r2_score") else "Correlation values"
        ),
    )
    ax.set_ylabel("Number of voxels")

    if scoring_metric is r2_score:
        ax.set_title(f"Histogram of explainable variance and $R^2$ for {sub_name}")
    else:
        ax.set_title(
            f"Histogram of explainable variance and correlation for {sub_name}"
        )

    ax.grid("on")
    ax.legend()
    return ax


def explainable_variance(y_matrix, bias_correction=True, do_zscore=True):
    """
    Adapted from gallantlab/himalaya
    BSD 3-Clause License
    Copyright (c) 2020, the himalaya developers All rights reserved.

    Compute explainable variance for a set of voxels.

    Parameters
    ----------
    y_matrix : array of shape (n_repeats * n_stimuli, n_voxels)
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
    n_repeats = 3  # NOTE : Hard-coded for THINGS dataset
    n_stimuli, n_voxels = y_matrix.shape
    data = y_matrix.reshape((n_stimuli // n_repeats, n_repeats, n_voxels)).swapaxes(
        0, 1
    )

    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    expl_var = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        expl_var = expl_var - (1 - expl_var) / (n_repeats - 1)
    return expl_var


def ridgeCV_sklearn(X_matrix, y_matrix, groups=None, scoring=r2_score):
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
    from sklearn.linear_model import RidgeCV

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit_transform(X_matrix)
    scaler.fit_transform(y_matrix)

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        outer_cv = GroupKFold(shuffle=True, random_state=0)
    alphas = np.logspace(1, 20, 20)
    estimator = RidgeCV(
        alphas=alphas,
        alpha_per_target=True,
        cv=None,
    )
    scorer = make_scorer(scoring)
    sklearn.set_config(enable_metadata_routing=True)

    scores = cross_validate(
        estimator,
        X_matrix,
        y=y_matrix,
        cv=outer_cv,
        scoring=scorer,
        params={"groups": groups} if groups is not None else None,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    return scores


def ridgeCV_himalaya(X_matrix, y_matrix, groups=None, scoring=r2_score):
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
    from himalaya.ridge import RidgeCV
    from himalaya.backend import set_backend

    backend = set_backend("torch_cuda", on_error="warn")

    scores = defaultdict()
    train_indices, test_indices = [], []
    best_scores = []
    best_alphas = []

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        outer_cv = GroupKFold(shuffle=True, random_state=0)

    alphas = np.logspace(1, 20, 20)
    pl = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        RidgeCV(
            alphas=alphas,
            solver_params=dict(
                n_targets_batch=500, n_alphas_batch=5, n_targets_batch_refit=100
            ),
        ),
    )

    for train_index, test_index in outer_cv.split(X_matrix, y_matrix, groups):

        train_indices.append(train_index)
        test_indices.append(test_index)

        pl.fit(X_matrix[train_index], y_matrix[train_index])

        if scoring is correlation_score:
            y_pred = pl.predict(X_matrix[test_index])
            best_scores.append(correlation_score(y_matrix[test_index], y_pred))
        else:
            best_scores.append(pl.score(X_matrix[test_index], y_matrix[test_index]))

        best_alphas.append(pl[-1].best_alphas_)

    scores["best_alphas"] = best_alphas
    scores["best_scores"] = best_scores
    scores["indices"] = {"train": train_indices, "test": test_indices}

    return scores


@click.command()
@click.option("--sub_name", default="sub-01", help="Subject name.")
@click.option("--roi", default=None, help="Region-of-interest")
@click.option("--cv_strategy", default="image", help="Cross-validation strategy")
@click.option(
    "--scoring_metric",
    default="r2_score",
    help="Desired scoring metric. Currently only 'r2_score' and 'correlation_score' "
    "are supported.",
)
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
def main(sub_name, roi, cv_strategy, scoring_metric, average, data_dir, engine):
    """ """
    rois = [None, "EBA", "FFA", "OFA", "pSTS", "MPA", "OPA", "PPA"]
    if roi not in rois:
        err_msg = f"Unrecognized ROI {roi}"
        raise ValueError(err_msg)

    sub_names = ["sub-01", "sub-02", "sub-03", "sub-06"]
    if sub_name not in sub_names:
        err_msg = f"Unrecognized subject {sub_name}"
        raise ValueError(err_msg)

    cv_strategies = ["image", "category", "kfold"]
    if cv_strategy not in cv_strategies:
        err_msg = f"Unrecognized cross-validation strategy {cv_strategy}"
        raise ValueError(err_msg)

    if average and (cv_strategy == "image"):
        err_msg = "Cross-validation strategy 'image' is not compatible with 'average'"
        raise ValueError(err_msg)

    scoring_metrics = ["r2_score", "correlation_score"]
    if scoring_metric not in scoring_metrics:
        err_msg = f"Unrecognized scoring metric {scoring_metric}"
        raise ValueError(err_msg)

    engines = ["himalaya", "sklearn"]
    if engine not in engines:
        err_msg = f"Unrecognized engine {engine}"
        raise ValueError(err_msg)

    if scoring_metric == "r2_score":
        scoring = r2_score
    if scoring_metric == "correlation_score":
        scoring = correlation_score

    # TODO: Remove this when tested
    if roi is not None:
        raise NotImplementedError

    if cv_strategy == "kfold":
        groups = None
    else:
        groups = np.loadtxt(
            Path(data_dir, f"{sub_name}_{cv_strategy}_outerCV_groups.txt"),
            dtype=np.str_,
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
        if groups is not None:
            groups = groups[::3]
        X_matrix = X_matrix[::3]
        y_matrix = np.mean(
            y_matrix.reshape(len(X_matrix), 3, y_matrix.shape[-1]), axis=1
        )

    if engine == "sklearn":
        scores = ridgeCV_sklearn(X_matrix, y_matrix, groups=groups, scoring=scoring)
        best_alphas = [estim.alpha_ for estim in scores["estimator"]]
        best_scores = [estim.best_score_ for estim in scores["estimator"]]
    elif engine == "himalaya":
        scores = ridgeCV_himalaya(X_matrix, y_matrix, groups=groups, scoring=scoring)
        best_alphas = scores["best_alphas"]
        best_scores = scores["best_scores"]

    np.save(f"{sub_name}_cv-{cv_strategy}_r2_scores_clip.npy")

    expl_var = explainable_variance(y_matrix)
    fig_hist = plot_voxel_hist(
        sub_name, expl_var, scores["cv_results"], scoring_metric=r2_score
    )
    fig_hist.savefig(f"{sub_name}_{cv_strategy}_{scoring_metric}_expl_var_hist.png")

    fig_alphas, ax = plt.subplots(1, 1)
    for i, b_alpha in enumerate(best_alphas):
        plot_alphas_diagnostic(
            best_alphas=b_alpha, alphas=np.logspace(1, 20, 20), cv_fold=i, ax=ax
        )
    fig_alphas.savefig(f"{sub_name}_{cv_strategy}_alphas.png")

    # fig_flat = plot_flatmap(
    #     best_scores, sub_name, mask_img, cv_strategy, scoring_metric="r2_score"
    # )


if __name__ == "__main__":
    main()
