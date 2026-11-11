
import pandas as pd
import pytest

from mineral_id.splits import make_grouped_splits


def create_test_metadata():

    records = []

    for mineral in ["Mineral_A", "Mineral_B", "Mineral_C"]:

        for i in range(10):

            sample_id = f"{mineral}_{i}"

            records.append({
                "rruff_id": sample_id,
                "mineral": mineral,
                "filename": f"{sample_id}_532.txt"
            })

    # Add a second spectrum from an existing sample.
    # Both measurements must remain in the same partition.
    records.append({
        "rruff_id": "Mineral_A_0",
        "mineral": "Mineral_A",
        "filename": "Mineral_A_0_second.txt"
    })

    return pd.DataFrame(records)


def test_grouped_splits_keep_sample_ids_together():

    metadata = create_test_metadata()

    splits = make_grouped_splits(
        metadata,
        random_state=42
    )

    train_ids = set(splits["train"]["rruff_id"])
    calibration_ids = set(splits["calibration"]["rruff_id"])
    test_ids = set(splits["test"]["rruff_id"])

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)


def test_all_classes_appear_in_each_split():

    metadata = create_test_metadata()

    splits = make_grouped_splits(metadata)

    original_classes = set(metadata["mineral"])

    for split_data in splits.values():

        assert set(split_data["mineral"]) == original_classes


def test_splits_are_reproducible():

    metadata = create_test_metadata()

    first = make_grouped_splits(
        metadata,
        random_state=42
    )

    second = make_grouped_splits(
        metadata,
        random_state=42
    )

    for name in first:

        assert set(first[name]["rruff_id"]) == set(
            second[name]["rruff_id"]
        )


def test_conflicting_group_labels_raise_error():

    metadata = pd.DataFrame({

        "rruff_id": [
            "R000001",
            "R000001",
            "R000002",
            "R000003"
        ],

        "mineral": [
            "Quartz",
            "Diamond",
            "Quartz",
            "Diamond"
        ]
    })

    with pytest.raises(ValueError):

        make_grouped_splits(metadata)
