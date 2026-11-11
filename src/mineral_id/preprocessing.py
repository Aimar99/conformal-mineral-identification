
from __future__ import annotations

import numpy as np
import pandas as pd


def make_raman_grid(
    start: float = 205.0,
    stop: float = 1200.0,
    step: float = 0.5
) -> np.ndarray:
    """
    Construct a uniform Raman-shift grid.

    Parameters
    ----------
    start : float
        First Raman-shift position (cm^-1).
    stop : float
        Last Raman-shift position (cm^-1).
    step : float
        Grid spacing (cm^-1).

    Returns
    -------
    np.ndarray
        Uniformly spaced Raman-shift positions.
    """

    if not np.isfinite([start, stop, step]).all():
        raise ValueError("Grid parameters must be finite.")

    if stop <= start:
        raise ValueError("'stop' must be greater than 'start'.")

    if step <= 0:
        raise ValueError("'step' must be positive.")

    n_intervals_float = (stop - start) / step
    n_intervals = round(n_intervals_float)

    # Ensure that the interval is divisible by the step.
    if not np.isclose(
        n_intervals_float,
        n_intervals,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(
            "The grid interval must be divisible by the step."
        )

    grid = start + step * np.arange(
        n_intervals + 1,
        dtype=float
    )

    # Avoid small floating-point endpoint discrepancies.
    grid[-1] = stop

    return grid


def interpolate_spectrum(
    spectrum: pd.DataFrame,
    grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate a Raman spectrum onto a common grid.

    Linear interpolation is used.

    Extrapolation is not permitted. The original Raman-shift
    positions must also be strictly increasing.

    Parameters
    ----------
    spectrum : pd.DataFrame
        Columns:
        - raman_shift_cm1
        - intensity

    grid : np.ndarray
        Target Raman-shift positions.

    Returns
    -------
    np.ndarray
        Interpolated intensity values.
    """

    required_columns = {
        "raman_shift_cm1",
        "intensity"
    }

    if not required_columns.issubset(spectrum.columns):
        raise ValueError(
            f"Spectrum must contain {required_columns}."
        )

    x = spectrum["raman_shift_cm1"].to_numpy(dtype=float)
    y = spectrum["intensity"].to_numpy(dtype=float)

    grid = np.asarray(grid, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or grid.ndim != 1:
        raise ValueError("All input arrays must be one-dimensional.")

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Invalid spectrum dimensions.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spectrum contains non-finite values.")

    if not np.isfinite(grid).all() or len(grid) < 2:
        raise ValueError("Invalid interpolation grid.")

    if np.any(np.diff(x) <= 0):
        raise ValueError(
            "Original Raman-shift positions must be "
            "strictly increasing and contain no duplicates."
        )

    if np.any(np.diff(grid) <= 0):
        raise ValueError("Target grid must be strictly increasing.")

    tolerance = 1e-9

    # Prevent extrapolation.
    if (
        grid[0] < x[0] - tolerance
        or grid[-1] > x[-1] + tolerance
    ):
        raise ValueError(
            "Target grid extends outside the original spectrum."
        )

    interpolated = np.interp(grid, x, y)

    return interpolated


def integrated_area(
    intensity: np.ndarray,
    grid: np.ndarray
) -> float:
    """
    Calculate the signed area under a spectrum using
    the composite trapezoidal rule.
    """

    intensity = np.asarray(intensity, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if intensity.ndim != 1 or grid.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(intensity) != len(grid) or len(grid) < 2:
        raise ValueError("Intensity and grid dimensions must match.")

    if not np.isfinite(intensity).all():
        raise ValueError("Intensity contains non-finite values.")

    if not np.isfinite(grid).all():
        raise ValueError("Grid contains non-finite values.")

    if np.any(np.diff(grid) <= 0):
        raise ValueError("Grid must be strictly increasing.")

    intervals = np.diff(grid)

    area = np.sum(
        0.5 * (intensity[:-1] + intensity[1:]) * intervals
    )

    return float(area)


def normalize_spectrum(
    intensity: np.ndarray,
    method: str = "l2",
    grid: np.ndarray | None = None,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Normalize spectral intensity.

    Supported methods
    -----------------
    max:
        Divide by maximum positive intensity.

    l2:
        Divide by the Euclidean norm.

    area:
        Divide by the signed integrated spectral area.

    Negative intensities are preserved.
    """

    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 1:
        raise ValueError("Intensity must be one-dimensional.")

    if not np.isfinite(intensity).all():
        raise ValueError("Intensity contains non-finite values.")

    if eps <= 0:
        raise ValueError("'eps' must be positive.")

    method = method.lower()

    if method == "max":

        scale = np.max(intensity)

    elif method == "l2":

        scale = np.linalg.norm(intensity)

    elif method == "area":

        if grid is None:
            raise ValueError(
                "A Raman-shift grid is required for area normalization."
            )

        scale = integrated_area(intensity, grid)

    else:

        raise ValueError(
            "Unknown normalization method. "
            "Choose 'max', 'l2' or 'area'."
        )

    if not np.isfinite(scale) or scale <= eps:
        raise ValueError(
            f"Invalid normalization scale for method '{method}'."
        )

    return intensity / scale


def preprocess_spectrum(
    spectrum: pd.DataFrame,
    grid: np.ndarray,
    normalization: str = "l2"
) -> np.ndarray:
    """
    Interpolate and normalize one Raman spectrum.

    Returns
    -------
    np.ndarray
        Model-ready intensity vector.
    """

    interpolated = interpolate_spectrum(
        spectrum=spectrum,
        grid=grid
    )

    normalized = normalize_spectrum(
        intensity=interpolated,
        method=normalization,
        grid=grid
    )

    return normalized
