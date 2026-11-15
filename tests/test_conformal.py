
import numpy as np
import pytest

from mineral_id.conformal import (
    calibration_scores,
    finite_sample_quantile,
    prediction_sets
)


def test_calibration_scores():

    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7]
    ])

    y_true = np.array(["A", "B"])

    classes = np.array(["A", "B"])

    scores = calibration_scores(
        probabilities,
        y_true,
        classes
    )

    assert np.allclose(scores, [0.2, 0.3])


def test_finite_sample_quantile_90():

    scores = np.arange(1, 28) / 28

    threshold = finite_sample_quantile(
        scores,
        alpha=0.10
    )

    assert np.isclose(threshold, scores[25])


def test_finite_sample_quantile_95():

    scores = np.arange(1, 28) / 28

    threshold = finite_sample_quantile(
        scores,
        alpha=0.05
    )

    assert np.isclose(threshold, scores[26])


def test_prediction_sets():

    probabilities = np.array([
        [0.60, 0.35, 0.05],
        [0.10, 0.80, 0.10]
    ])

    classes = np.array(["A", "B", "C"])

    # Threshold 0.5 means probabilities >= 0.5
    sets = prediction_sets(
        probabilities,
        classes,
        threshold=0.5
    )

    assert sets == [["A"], ["B"]]


def test_empty_prediction_set_is_allowed():

    probabilities = np.array([
        [0.40, 0.35, 0.25]
    ])

    classes = np.array(["A", "B", "C"])

    sets = prediction_sets(
        probabilities,
        classes,
        threshold=0.4
    )

    assert sets == [[]]


def test_invalid_alpha_raises_error():

    with pytest.raises(ValueError):

        finite_sample_quantile(
            np.array([0.1, 0.2]),
            alpha=1.0
        )


def test_unknown_class_raises_error():

    probabilities = np.array([
        [0.8, 0.2]
    ])

    with pytest.raises(ValueError):

        calibration_scores(
            probabilities,
            np.array(["Unknown"]),
            np.array(["A", "B"])
        )
