
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def make_grouped_splits(
    metadata: pd.DataFrame,
    group_col: str = "rruff_id",
    label_col: str = "mineral",
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Create stratified train, calibration and test partitions.

    All observations belonging to the same group are kept
    in exactly one partition.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata describing each spectral observation.
    group_col : str
        Column identifying the RRUFF sample group.
    label_col : str
        Column containing the mineral class.
    train_fraction : float
        Fraction allocated to training.
    calibration_fraction : float
        Fraction allocated to conformal calibration.
    test_fraction : float
        Fraction allocated to final testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - train
        - calibration
        - test
    """

    # --------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------

    required_columns = {group_col, label_col}

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if metadata.empty:
        raise ValueError("Input metadata is empty.")

    if metadata[[group_col, label_col]].isna().any().any():
        raise ValueError(
            "Group identifiers and labels cannot contain missing values."
        )

    fractions = [
        train_fraction,
        calibration_fraction,
        test_fraction
    ]

    if any(f <= 0 or f >= 1 for f in fractions):
        raise ValueError("All split fractions must be between 0 and 1.")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    # --------------------------------------------------
    # 2. Check that each group has only one mineral label
    # --------------------------------------------------

    labels_per_group = (
        metadata.groupby(group_col)[label_col].nunique()
    )

    inconsistent_groups = labels_per_group[
        labels_per_group > 1
    ]

    if not inconsistent_groups.empty:
        raise ValueError(
            "Some sample groups have multiple labels: "
            f"{inconsistent_groups.index.tolist()}"
        )

    # One row per unique group
    groups = (
        metadata[[group_col, label_col]]
        .drop_duplicates()
        .sort_values([label_col, group_col])
        .reset_index(drop=True)
    )

    # Check sufficient samples per class
    class_counts = groups[label_col].value_counts()

    if class_counts.min() < 5:
        raise ValueError(
            "At least 5 sample groups per class are required "
            "for this splitting procedure."
        )

    # --------------------------------------------------
    # 3. Create training and temporary holdout sets
    # --------------------------------------------------

    holdout_fraction = (
        calibration_fraction + test_fraction
    )

    train_groups, holdout_groups = train_test_split(
        groups,
        test_size=holdout_fraction,
        stratify=groups[label_col],
        random_state=random_state
    )

    # --------------------------------------------------
    # 4. Divide holdout into calibration and test
    # --------------------------------------------------

    relative_test_fraction = (
        test_fraction / holdout_fraction
    )

    calibration_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=relative_test_fraction,
        stratify=holdout_groups[label_col],
        random_state=random_state + 1
    )

    # --------------------------------------------------
    # 5. Recover all spectral observations for each group
    # --------------------------------------------------

    split_groups = {
        "train": set(train_groups[group_col]),
        "calibration": set(calibration_groups[group_col]),
        "test": set(test_groups[group_col])
    }

    splits = {
        name: (
            metadata[
                metadata[group_col].isin(group_ids)
            ]
            .sort_values([label_col, group_col])
            .reset_index(drop=True)
        )
        for name, group_ids in split_groups.items()
    }

    # --------------------------------------------------
    # 6. Verify no group leakage
    # --------------------------------------------------

    train_ids = split_groups["train"]
    calibration_ids = split_groups["calibration"]
    test_ids = split_groups["test"]

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)

    all_original_ids = set(metadata[group_col])

    all_split_ids = (
        train_ids | calibration_ids | test_ids
    )

    assert all_original_ids == all_split_ids

    # Ensure all classes appear in all three partitions
    all_classes = set(metadata[label_col].unique())

    for name, split_data in splits.items():

        if set(split_data[label_col].unique()) != all_classes:
            raise RuntimeError(
                f"Some classes are missing from the {name} split."
            )

    return splits
