import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix


def pred_image_categories(sub_name, data_dir):
    """
    Predict 720 image categories from CLIP embeddings.
    Each image is assigned to only one category.

    Parameters
    ----------
    sub_name : str
    data_dir : str
    """
    stim_vec = np.loadtxt(
        Path(data_dir, "encoding-inputs", f"{sub_name}_stim_labels.txt"),
        dtype=np.str_,
    )
    X_matrix = np.load(
        Path(data_dir, "encoding-inputs", f"{sub_name}_stim_features.npy")
    )

    stim_cat = np.asarray([sv.rsplit("_", 1)[0] for sv in stim_vec])
    lb = LabelBinarizer().fit(stim_cat)
    cat_y = lb.transform(stim_cat)
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix, cat_y, test_size=0.33, random_state=2
    )

    clf = OneVsRestClassifier(LogisticRegression())
    # clf = LinearSVC()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

    cfm = multilabel_confusion_matrix(y_test, y_pred)

    # y_score = jaccard_score(y_test, y_pred, average="micro")

    return cfm, accuracy


def pred_THINGSPlus_categories(sub_name, data_dir):
    """
    Predict multi-label 53 THINGSPlus image categories from CLIP embeddings.
    Each image may be assigned to multiple categories.

    Parameters
    ----------
    sub_name : str
    data_dir : str
    """
    with open(Path(data_dir, "encoding-inputs", "category53_mapping.json")) as f:
        cat_dict = json.load(f)

    stim_vec = np.loadtxt(
        Path(data_dir, "encoding-inputs", f"{sub_name}_stim_labels.txt"),
        dtype=np.str_,
    )
    X_matrix = np.load(
        Path(data_dir, "encoding-inputs", f"{sub_name}_stim_features.npy")
    )

    cat53_stim_mask_ = [True if sv in cat_dict.keys() else False for sv in stim_vec]
    cat53_X = X_matrix[cat53_stim_mask_]

    cat53_dense_labels_ = []
    for sv in stim_vec[cat53_stim_mask_]:
        cat53_dense_labels_.append(cat_dict.get(sv))

    mlb = MultiLabelBinarizer().fit(cat53_dense_labels_)
    cat53_y = mlb.transform(cat53_dense_labels_)
    X_train, X_test, y_train, y_test = train_test_split(
        cat53_X, cat53_y, test_size=0.33, random_state=2
    )

    # run classification analysis
    clf = OneVsRestClassifier(LogisticRegression())
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    cfm = multilabel_confusion_matrix(y_test, y_score >= 0.5)

    ovr_score = jaccard_score(y_test, y_score >= 0.5, average="samples")

    # Compare with ensemble of binary classifiers
    # See : https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html#sphx-glr-auto-examples-multioutput-plot-classifier-chain-yeast-py
    chains = [
        ClassifierChain(LogisticRegression(), order="random", random_state=i)
        for i in range(10)
    ]
    for chain in chains:
        chain.fit(X_train, y_train)

    y_pred_chains = np.array([chain.predict_proba(X_test) for chain in chains])
    chain_scores = [
        jaccard_score(y_test, y_pred_chain >= 0.5, average="samples")
        for y_pred_chain in y_pred_chains
    ]

    Y_pred_ensemble = y_pred_chains.mean(axis=0)
    ensemble_score = jaccard_score(y_test, Y_pred_ensemble >= 0.5, average="samples")

    return cfm, ovr_score, chain_scores, ensemble_score


def gen_THINGSPlus_categories(sub_name, data_dir):
    """
    Parameters
    ----------
    sub_name : str
    data_dir : str
    """
    annot_fname = f"{sub_name}_task-things_desc-perTrial_annotation.tsv"

    annot_df = pd.read_csv(Path(data_dir, "annot", annot_fname), sep="\t")
    annot_df = annot_df.loc[annot_df["exclude_session"] == False]
    annot_df = annot_df.loc[annot_df["atypical"] == False]

    cat53_mask = annot_df[annot_df["highercat53_names"] != "[]"].index
    image_names = annot_df["image_name"][cat53_mask]

    cat53_names = annot_df["highercat53_names"][cat53_mask].str.replace(
        r"'|\]|\[", "", regex=True
    )
    sanitized_ = []
    for cat in cat53_names.values:
        list_labels = cat.split(",")
        sanitized_.append([l.strip() for l in list_labels])

    cat_dict = dict(zip(image_names, pd.Series(sanitized_)))
    # FIXME : this is consolidating duplicate keys, which is not what we want
    with open(
        Path(data_dir, "encoding-inputs", "category53_mapping.json"),
        "w",
        encoding="utf8",
    ) as outfile:
        json.dump(cat_dict, outfile)

    return cat_dict


def gen_THINGSPlus_groups(data_dir, data_dict=None):
    """
    Note
    ----
    The resulting train, test splits will be of unequal sizes ;
    that is, images that are not labelled "animal" may also be not labelled
    "breakfast food," and so assigned to the training split multiple times.
    The resulting distribution of labels is known to have a significant
    rightward-skew given the pre-existing label distribution (i.e., the category
    "animal" is more likely to occur overall).
    """
    if data_dict is None:
        # NOTE, FIXME : this is consolidating duplicate keys, which is not what we want
        with open(
            Path(data_dir, "encoding-inputs", "category53_mapping.json"),
            "r",
            encoding="utf8",
        ) as infile:
            data_dict = json.load(infile)

    classes = list(data_dict.values())
    unique_classes = np.unique(np.concatenate(classes).ravel())

    split_train, split_test = [], []

    for _, unique in enumerate(unique_classes):
        train, test = [], []

        for s in classes:
            if unique in s:
                test.append(s)
            else:
                train.append(s)

        split_train.append(train)
        split_test.append(test)

    # group_mapping = dict(zip(data_dict.keys(), group_labels))
    return
