import random

import pytest

from nucleus import DatasetItem, deduplicate_by_phash
from nucleus.local_deduplication import LocalDeduplicationResult


def _item(reference_id, phash):
    return DatasetItem(
        image_location=f"https://example.com/{reference_id}.jpg",
        reference_id=reference_id,
        phash=phash,
    )


def _row(reference_id, phash):
    return {
        "item": _item(reference_id, phash),
        "annotations": {"box": [f"annotation-{reference_id}"]},
    }


def _item_without_reference_id(phash):
    item = DatasetItem.__new__(DatasetItem)
    item.reference_id = None
    item.phash = phash
    return item


def _flip_bits(phash, indexes):
    bits = list(phash)
    for index in indexes:
        bits[index] = "1" if bits[index] == "0" else "0"
    return "".join(bits)


def test_deduplicate_by_phash_preserves_generator_rows():
    base = "0" * 64
    near = _flip_bits(base, [63])
    far = "1" * 64
    row_b = _row("b", near)
    row_a = _row("a", base)
    row_c = _row("c", far)

    result = deduplicate_by_phash([row_b, row_a, row_c], threshold=1)

    assert isinstance(result, LocalDeduplicationResult)
    assert result.unique == [row_a, row_c]
    assert result.unique_dataset_items == [row_a["item"], row_c["item"]]
    assert result.unique_reference_ids == ["a", "c"]
    assert result.stats.threshold == 1
    assert result.stats.original_count == 3
    assert result.stats.deduplicated_count == 2


def test_deduplicate_by_phash_accepts_dataset_items():
    base = "0" * 64
    duplicate = "0" * 64
    distinct = _flip_bits(base, [63])
    item_a = _item("a", base)
    item_b = _item("b", duplicate)
    item_c = _item("c", distinct)

    result = deduplicate_by_phash([item_b, item_c, item_a], threshold=0)

    assert result.unique == [item_a, item_c]
    assert result.unique_dataset_items == [item_a, item_c]
    assert result.unique_reference_ids == ["a", "c"]
    assert result.stats.original_count == 3
    assert result.stats.deduplicated_count == 2


def test_deduplicate_by_phash_threshold_ten_uses_hamming_distance():
    base = "0" * 64
    distance_ten = _flip_bits(base, range(10))
    distance_eleven = _flip_bits(base, range(11))
    row_a = _row("a", base)
    row_b = _row("b", distance_ten)
    row_c = _row("c", distance_eleven)

    result = deduplicate_by_phash([row_c, row_b, row_a], threshold=10)

    assert result.unique == [row_a, row_c]
    assert result.unique_reference_ids == ["a", "c"]
    assert result.stats.deduplicated_count == 2


@pytest.mark.parametrize("threshold", [1, 5, 10, 11, 20])
def test_deduplicate_by_phash_matches_linear_baseline(threshold):
    random.seed(123)
    rows = [
        _row(f"item_{i:03d}", f"{random.getrandbits(64):064b}")
        for i in range(200)
    ]

    baseline_unique_reference_ids = []
    baseline_kept_phashes = []
    for row in sorted(
        rows,
        key=lambda candidate: (
            candidate["item"].phash,
            candidate["item"].reference_id,
        ),
    ):
        phash_value = int(row["item"].phash, 2)
        if any(
            (phash_value ^ kept_phash).bit_count() <= threshold
            for kept_phash in baseline_kept_phashes
        ):
            continue
        baseline_kept_phashes.append(phash_value)
        baseline_unique_reference_ids.append(row["item"].reference_id)

    result = deduplicate_by_phash(rows, threshold=threshold)

    assert result.unique_reference_ids == baseline_unique_reference_ids


def test_deduplicate_by_phash_threshold_sixty_four_keeps_one_item():
    row_a = _row("a", "1" * 64)
    row_b = _row("b", "0" * 64)
    row_c = _row("c", _flip_bits("0" * 64, [63]))

    result = deduplicate_by_phash([row_a, row_c, row_b], threshold=64)

    assert result.unique == [row_b]
    assert result.unique_reference_ids == ["b"]
    assert result.stats.original_count == 3
    assert result.stats.deduplicated_count == 1


def test_deduplicate_by_phash_empty_input_returns_empty_result():
    result = deduplicate_by_phash([], threshold=10)

    assert result.unique == []
    assert result.unique_dataset_items == []
    assert result.unique_reference_ids == []
    assert result.stats.threshold == 10
    assert result.stats.original_count == 0
    assert result.stats.deduplicated_count == 0


@pytest.mark.parametrize("threshold", [-1, 65, 1.5, True])
def test_deduplicate_by_phash_rejects_invalid_threshold(threshold):
    with pytest.raises(
        ValueError,
        match="threshold must be an integer between 0 and 64",
    ):
        deduplicate_by_phash([], threshold=threshold)


@pytest.mark.parametrize("phash", [None, "", "0" * 63, "0" * 65, "x" * 64])
def test_deduplicate_by_phash_rejects_missing_or_malformed_phash(phash):
    with pytest.raises(
        ValueError,
        match="All DatasetItems must have a 64-character binary pHash",
    ):
        deduplicate_by_phash([_item("bad", phash)], threshold=10)


def test_deduplicate_by_phash_rejects_unsupported_input_shape():
    with pytest.raises(
        TypeError,
        match="Expected each object to be a DatasetItem",
    ):
        deduplicate_by_phash([{"not_item": _item("a", "0" * 64)}], threshold=0)


def test_deduplicate_by_phash_uses_custom_sort_key():
    row_a = _row("a", "0" * 64)
    row_b = _row("b", "0" * 64)

    result = deduplicate_by_phash(
        [row_a, row_b],
        threshold=0,
        sort_key=lambda row: 0 if row["item"].reference_id == "b" else 1,
    )

    assert result.unique == [row_b]
    assert result.unique_reference_ids == ["b"]


def test_deduplicate_by_phash_sorts_integer_sort_key_numerically():
    row_a = _row("a", "0" * 64)
    row_b = _row("b", "0" * 64)

    result = deduplicate_by_phash(
        [row_a, row_b],
        threshold=0,
        sort_key=lambda row: 10 if row["item"].reference_id == "a" else 9,
    )

    assert result.unique == [row_b]
    assert result.unique_reference_ids == ["b"]


def test_deduplicate_by_phash_preserves_ordinal_fallback_order():
    rows = [
        _row(
            f"item_{index}", "1" * 64 if index in (2, 10) else f"{index:064b}"
        )
        for index in range(11)
    ]

    result = deduplicate_by_phash(
        rows,
        threshold=0,
        sort_key=lambda row: None,
    )

    assert "item_2" in result.unique_reference_ids
    assert "item_10" not in result.unique_reference_ids


def test_deduplicate_by_phash_preserves_missing_reference_ids():
    item_without_reference_id = _item_without_reference_id("0" * 64)
    item_with_reference_id = _item("b", "1" * 64)

    result = deduplicate_by_phash(
        [item_with_reference_id, item_without_reference_id],
        threshold=0,
    )

    assert result.unique_reference_ids == [None, "b"]
