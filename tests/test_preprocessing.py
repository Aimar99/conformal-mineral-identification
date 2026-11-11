
import numpy as np
import pandas as pd
import pytest

from mineral_id.preprocessing import (
    make_raman_grid,
    interpolate_spectrum,
    integrated_area,
    normalize_spectrum,
    preprocess_spectrum
)


def test_make_raman_grid():

    grid = make_raman_grid(205, 1200, 0.5)

    assert len(grid) == 1991
    assert np.isclose(grid[0], 205)
    assert np.isclose(grid[-1], 1200)
    assert np.allclose(np.diff(grid), 0.5)


def test_invalid_grid_step_raises_error():

    with pytest.raises(ValueError):

        make_raman_grid(205, 1200, 0.3)


def test_linear_interpolation():

    spectrum = pd.DataFrame({
        "raman_shift_cm1": [0, 1, 2],
        "intensity": [1, 3, 5]
    })

    grid = np.array([0.5, 1.5])

    result = interpolate_spectrum(spectrum, grid)

    assert np.allclose(result, [2, 4])


def test_extrapolation_is_rejected():

    spectrum = pd.DataFrame({
        "raman_shift_cm1": [0, 1, 2],
        "intensity": [1, 3, 5]
    })

    grid = np.array([-0.5, 1, 2])

    with pytest.raises(ValueError):

        interpolate_spectrum(spectrum, grid)


def test_duplicate_shifts_are_rejected():

    spectrum = pd.DataFrame({
        "raman_shift_cm1": [0, 1, 1, 2],
        "intensity": [1, 2, 3, 4]
    })

    grid = np.array([0.5, 1.5])

    with pytest.raises(ValueError):

        interpolate_spectrum(spectrum, grid)


def test_max_normalization_preserves_negative_values():

    intensity = np.array([-1, 2, 1])

    result = normalize_spectrum(
        intensity,
        method="max"
    )

    assert np.allclose(result, [-0.5, 1, 0.5])


def test_l2_normalization():

    intensity = np.array([-3, 4])

    result = normalize_spectrum(
        intensity,
        method="l2"
    )

    assert np.isclose(np.linalg.norm(result), 1)


def test_area_normalization():

    grid = np.array([0, 1, 2])
    intensity = np.array([1, 2, 1])

    result = normalize_spectrum(
        intensity,
        method="area",
        grid=grid
    )

    assert np.isclose(
        integrated_area(result, grid),
        1
    )


def test_nonpositive_area_is_rejected():

    grid = np.array([0, 1, 2])
    intensity = np.array([-1, -2, -1])

    with pytest.raises(ValueError):

        normalize_spectrum(
            intensity,
            method="area",
            grid=grid
        )


def test_preprocess_spectrum():

    spectrum = pd.DataFrame({
        "raman_shift_cm1": [205, 206, 207],
        "intensity": [1, 2, 3]
    })

    grid = np.array([205, 205.5, 206, 206.5, 207])

    result = preprocess_spectrum(
        spectrum,
        grid,
        normalization="l2"
    )

    assert len(result) == len(grid)
    assert np.isclose(np.linalg.norm(result), 1)
