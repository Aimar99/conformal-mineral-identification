
from pathlib import Path

import pytest

from mineral_id.io import (
    read_rruff_metadata,
    read_rruff_spectrum
)


def test_read_rruff_metadata(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "##NAMES=Quartz\n"
        "##RRUFFID=R000001\n"
        "##RAMAN WAVELENGTH=532\n"
        "##FILETYPE=Raman Processed\n"
        "50.0, 1.0\n"
        "51.0, 2.0\n",
        encoding="utf-8"
    )

    metadata = read_rruff_metadata(file_path)

    assert metadata["NAMES"] == "Quartz"
    assert metadata["RRUFFID"] == "R000001"
    assert metadata["RAMAN WAVELENGTH"] == "532"


def test_read_rruff_spectrum(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "##NAMES=Quartz\n"
        "##RRUFFID=R000001\n"
        "50.0, 1.0\n"
        "51.0, 2.0\n"
        "52.0, 3.0\n",
        encoding="utf-8"
    )

    spectrum = read_rruff_spectrum(file_path)

    assert len(spectrum) == 3

    assert list(spectrum.columns) == [
        "raman_shift_cm1",
        "intensity"
    ]

    assert spectrum["raman_shift_cm1"].tolist() == [
        50.0, 51.0, 52.0
    ]


def test_repeated_raman_shifts_are_preserved(tmp_path):

    file_path = tmp_path / "repeated.txt"

    file_path.write_text(
        "##NAMES=Heflikite\n"
        "50.0, 1.0\n"
        "50.0, 2.0\n"
        "51.0, 3.0\n",
        encoding="utf-8"
    )

    spectrum = read_rruff_spectrum(file_path)

    assert len(spectrum) == 3

    assert (
        spectrum["raman_shift_cm1"]
        .duplicated()
        .sum()
        == 1
    )


def test_empty_spectrum_raises_error(tmp_path):

    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "##NAMES=Quartz\n"
        "##RRUFFID=R000001\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        read_rruff_spectrum(file_path)
