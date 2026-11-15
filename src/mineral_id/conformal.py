
from __future__ import annotations

import math
import numpy as np


def calibration_scores(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray
) -> np.ndarray:
    """
    Calculate inverse-probability nonconformity scores.

    Score:
        s(x, y) = 1 - P(Y=y | X=x)

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (n_samples, n_classes).
    y_true : np.ndarray
        True class label for each calibration observation.
    classes : np.ndarray
        Class ordering corresponding to probability columns.

    Returns
    -------
    np.ndarray
        One nonconformity score per calibration observation.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be two-dimensional.")

    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Inconsistent probability dimensions.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must lie between 0 and 1.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one.")

    class_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    unknown_labels = set(y_true) - set(classes)

    if unknown_labels:
        raise ValueError(
            f"Unknown calibration labels: {unknown_labels}"
        )

    true_class_indices = np.array([
        class_to_index[label]
        for label in y_true
    ])

    true_probabilities = probabilities[
        np.arange(len(y_true)),
        true_class_indices
    ]

    return 1.0 - true_probabilities


def finite_sample_quantile(
    scores: np.ndarray,
    alpha: float
) -> float:
    """
    Calculate the split-conformal finite-sample quantile.

    Uses:
        k = ceil((n + 1) * (1 - alpha))

    If k > n, the threshold is infinity.
    """

    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Scores must be a non-empty 1D array.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    n = len(scores)

    k = math.ceil((n + 1) * (1 - alpha))

    if k > n:
        return float("inf")

    sorted_scores = np.sort(scores)

    return float(sorted_scores[k - 1])


def prediction_sets(
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float
) -> list[list]:
    """
    Construct conformal prediction sets.

    Include class y when:
        1 - P(Y=y | X=x) <= threshold
    """

    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be two-dimensional.")

    if probabilities.shape[1] != len(classes):
        raise ValueError("Class count does not match probability columns.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must lie between 0 and 1.")

    if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("Probability rows must sum to one.")

    membership = (1.0 - probabilities) <= threshold

    return [
        classes[row_mask].tolist()
        for row_mask in membership
    ]
