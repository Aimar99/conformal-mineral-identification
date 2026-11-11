
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def read_rruff_metadata(file_path: str | Path) -> dict[str, str]:
    """
    Read metadata from a RRUFF spectral file.

    Parameters
    ----------
    file_path : str or Path
        Path to a RRUFF text file.

    Returns
    -------
    dict
        Dictionary containing the metadata key-value pairs.

    Notes
    -----
    Metadata lines begin with ##. Reading stops when
    the numerical spectral data section begins.
    """

    file_path = Path(file_path)
    metadata = {}

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if not line.startswith("##"):
                break

            content = line[2:]

            if "=" in content:
                key, value = content.split("=", 1)

                metadata[key.strip()] = value.strip()

    return metadata


def read_rruff_spectrum(file_path: str | Path) -> pd.DataFrame:
    """
    Read numerical Raman measurements from a RRUFF file.

    Parameters
    ----------
    file_path : str or Path
        Path to a processed Raman spectrum.

    Returns
    -------
    pandas.DataFrame
        Two columns:
        - raman_shift_cm1
        - intensity

    Notes
    -----
    Repeated Raman-shift positions are deliberately preserved.
    We do not sort or aggregate observations at this stage.
    """

    file_path = Path(file_path)

    try:
        spectrum = pd.read_csv(
            file_path,
            sep=",",
            comment="#",
            header=None,
            names=["raman_shift_cm1", "intensity"],
            skipinitialspace=True,
            engine="python",
            encoding="utf-8-sig"
        )

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"No numerical spectral data found in {file_path}"
        ) from error

    if spectrum.empty:
        raise ValueError(
            f"No numerical spectral data found in {file_path}"
        )

    # Convert both columns to numerical values.
    for column in ["raman_shift_cm1", "intensity"]:

        spectrum[column] = pd.to_numeric(
            spectrum[column],
            errors="coerce"
        )

    # Identify invalid or non-finite observations.
    values = spectrum[
        ["raman_shift_cm1", "intensity"]
    ].to_numpy(dtype=float)

    valid_rows = np.isfinite(values).all(axis=1)

    n_invalid = int((~valid_rows).sum())

    if n_invalid > 0:

        warnings.warn(
            f"Removed {n_invalid} invalid rows from {file_path.name}",
            RuntimeWarning
        )

        spectrum = spectrum.loc[valid_rows].copy()

    if len(spectrum) < 2:
        raise ValueError(
            f"Insufficient valid spectral observations in {file_path}"
        )

    return spectrum.reset_index(drop=True)
