import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target


def gen_thingsPlus_groups(data_dict=None):
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
        with open("category53_mapping.json", "r", encoding="utf8") as infile:
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

    group_mapping = dict(zip(data_dict.keys(), group_labels))
    return


def gen_thingsPlus_categories(sub_name, data_dir):
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
    with open("category53_mapping.json", "w", encoding="utf8") as outfile:
        json.dump(cat_dict, outfile)

    return cat_dict


class IterativeStratifiedKFold:

    def _make_test_folds_multi_label(self, y):
        """Make test folds for multi-label `y`.
        Supported `y` types: multilabel-indicator.
        References:
        - Sechidis, Konstantinos; Tsoumakas, Grigorios; Vlahavas, Ioannis.
            "On the stratification of multi-label data." Machine Learning and
            Knowledge Discovery in Databases (2011), 145--158.
            http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
        - https://github.com/trent-b/iterative-stratification (BSD 3 clause)
        """
        rng = check_random_state(self.random_state)
        n_samples = y.shape[0]

        # Multilabel-indicator has at most two classes, so we encode the first
        # occurrence as the negative example and the other as the positive example
        first_occurrence = y[0, 0]
        y = np.where(y == first_occurrence, 0, 1)

        # Number of positive examples per label (will be used to keep track of the
        # number of remaining positive examples in iterative stratification later)
        remaining_per_label = y.sum(axis=0)

        min_groups = np.inf
        for n_pos in remaining_per_label:
            n_neg = n_samples - n_pos
            if n_pos < self.n_splits and n_neg < self.n_splits:
                raise ValueError(
                    "n_splits=%d cannot be greater than the number of members in each "
                    "class for each label." % (self.n_splits)
                )
            min_groups = min(min_groups, n_pos, n_neg)

        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y among all labels has only %d "
                "members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        indices = np.arange(n_samples)
        if self.shuffle:
            rng.shuffle(indices)
            y = y[indices]
        test_folds = np.empty(n_samples, dtype="i")

        # Desired number of examples per-fold (n_splits,) and desired number of
        # examples per-fold per-label (n_splits, n_labels)
        props = np.asarray([1 / self.n_splits] * self.n_splits)
        desired = n_samples * props
        desired_per_label = np.outer(props, y.sum(axis=0))

        # Keep track of unassigned examples
        n_unassigned = n_samples
        unassigned_mask = np.ones(n_samples, dtype=bool)

        while n_unassigned > 0:
            # Find the label with the fewest (but >=1) remaining positive examples;
            # get the unassigned examples that are positive for the selected label
            selected_label = np.argmin(
                np.where(remaining_per_label != 0, remaining_per_label, np.inf)
            )
            selected_indices = np.where(y[:, selected_label] & unassigned_mask)[0]

            if len(selected_indices) == 0:
                # If there are no unassigned examples positive for the selected label,
                # it means that there are no remaining positive examples for any label;
                # in this case we try to distribute the rest of examples evenly
                for i in np.where(unassigned_mask)[0]:
                    max_fold = np.argmax(desired)
                    test_folds[i] = max_fold
                    desired[max_fold] -= 1
                break

            for i in selected_indices:
                # Find the fold with the largest number of desired positive examples
                # for this label, breaking ties by proporitizing the fold with the
                # largest number of total desired examples; note that folds with
                # non-positive number of total desired examples are excluded from
                # consideration, otherwise we cannot satisfy that smallest and largest
                # test sizes differ by at most one
                label_desired = desired_per_label[:, selected_label]
                max_fold = max(
                    np.where(desired > 0)[0],
                    key=lambda fold_idx: (label_desired[fold_idx], desired[fold_idx]),
                )

                test_folds[i] = max_fold
                unassigned_mask[i] = False
                n_unassigned -= 1

                desired_per_label[max_fold] -= y[i]
                remaining_per_label -= y[i]
                desired[max_fold] -= 1

        if self.shuffle:
            return test_folds[np.argsort(indices)]
        else:
            return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)

        allowed_single_target_types = ("binary", "multiclass")
        allowed_multi_target_types = ("multilabel-indicator",)

        if type_of_target_y in allowed_single_target_types:
            test_folds = self._make_test_folds_single_label(y)
        elif type_of_target_y in allowed_multi_target_types:
            test_folds = self._make_test_folds_multi_label(y)
        else:
            raise ValueError(
                "Supported target types are: "
                f"{allowed_single_target_types + allowed_multi_target_types}. "
                f"Got {type_of_target_y!r} instead."
            )

        for i in range(self.n_splits):
            yield test_folds == i
