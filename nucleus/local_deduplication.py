"""Local pHash-based deduplication utilities."""

from __future__ import annotations

import importlib
import itertools
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union

from .constants import ITEM_KEY
from .dataset_item import DatasetItem
from .deduplication import DeduplicationStats

_NATIVE_DEDUP: Optional[Any]
try:
    _NATIVE_DEDUP = importlib.import_module("nucleus._native_dedup")
except ImportError:
    _NATIVE_DEDUP = None

InputT = TypeVar("InputT")
_TieBreakKey = tuple[int, Union[int, str]]

PHASH_REGEX = re.compile(r"^[01]{64}$")
DEDUP_THRESHOLD_MIN = 0
DEDUP_THRESHOLD_MAX = 64
LOCAL_INDEX_MAX_THRESHOLD = 11
PARTITION_COUNT = 2
INDEX_CHUNK_BITS = (11, 11, 11, 11, 10, 10)
INDEX_CHUNK_COUNT = len(INDEX_CHUNK_BITS)
PHASH_VALUE_MASK = (1 << 64) - 1
ROTATED_PARTITION_BITS = 8


@dataclass
class LocalDeduplicationResult(Generic[InputT]):
    """Output of a local pHash deduplication run.

    Attributes:
        unique: Input objects that survived deduplication. If you passed rows
            from ``items_and_annotation_generator()``, this contains those same
            row dictionaries. If you passed ``DatasetItem`` objects, it contains
            ``DatasetItem`` objects.
        unique_dataset_items: The DatasetItem for each object in ``unique``.
        unique_reference_ids: Reference IDs for the DatasetItems in ``unique``.
            Entries can be ``None`` if a DatasetItem has no reference ID.
        stats: Summary statistics for the run.
    """

    unique: List[InputT]
    unique_dataset_items: List[DatasetItem]
    unique_reference_ids: List[Optional[str]]
    stats: DeduplicationStats


@dataclass(slots=True)
class _DeduplicationRecord(Generic[InputT]):
    stable_id: _TieBreakKey
    phash_value: int
    obj: InputT
    dataset_item: DatasetItem


class _HammingIndex:
    """Exact Hamming near-neighbor index for 64-bit hashes.

    The first partition splits the pHash into six contiguous chunks.
    The second partition rotates the pHash by eight bits, then applies the same
    chunking. If two hashes are within threshold ``t``, then each partition has
    at least one chunk within ``floor(t / 6)``. Querying both partitions and
    intersecting the candidates keeps the result exact while avoiding a full
    scan for the intended ``t <= 11`` case.

    The chunk tolerance and the rotation are separate mechanisms. For
    threshold 10, ``floor(10 / 6) == 1``, so a query chunk ``x`` probes the
    exact bucket ``x`` plus the one-bit-away buckets ``x ^ (1 << bit_offset)``
    inside that chunk. The eight-bit rotation does not implement that
    tolerance; it gives a second chunk-boundary alignment so fewer false
    candidates survive to the exact full-hash check.
    """

    def __init__(self, threshold: int):
        self._threshold = threshold
        self._chunk_radius = threshold // INDEX_CHUNK_COUNT
        self._variant_masks_by_chunk = [
            _variant_masks(self._chunk_radius, chunk_bits)
            for chunk_bits in INDEX_CHUNK_BITS
        ]
        self._hashes: List[int] = []
        self._candidate_marks = bytearray()
        self._indexes: List[List[dict[int, List[int]]]] = [
            [{} for _ in range(INDEX_CHUNK_COUNT)]
            for _ in range(PARTITION_COUNT)
        ]

    def _add(self, phash_value: int) -> None:
        kept_index = len(self._hashes)
        self._hashes.append(phash_value)
        self._candidate_marks.append(0)
        for partition_index in range(PARTITION_COUNT):
            chunks = _partition_chunks(phash_value, partition_index)
            for chunk_index, chunk_value in enumerate(chunks):
                index = self._indexes[partition_index][chunk_index]
                index.setdefault(chunk_value, []).append(kept_index)

    def add_if_unique(self, phash_value: int) -> bool:
        if self._find_duplicate_index(phash_value) is not None:
            return False
        self._add(phash_value)
        return True

    def _find_duplicate_index(self, phash_value: int) -> Optional[int]:
        if not self._hashes:
            return None

        touched_indexes = self._mark_partition_zero_candidates(phash_value)
        if not touched_indexes:
            return None

        duplicate_index = self._find_marked_partition_one_duplicate(
            phash_value
        )
        _clear_candidate_marks(self._candidate_marks, touched_indexes)
        return duplicate_index

    def _mark_partition_zero_candidates(self, phash_value: int) -> List[int]:
        touched_indexes: List[int] = []
        marks = self._candidate_marks
        partition_zero_chunks = _partition_chunks(phash_value, 0)
        for chunk_index, chunk_value in enumerate(partition_zero_chunks):
            index = self._indexes[0][chunk_index]
            for mask in self._variant_masks_by_chunk[chunk_index]:
                bucket = index.get(chunk_value ^ mask)
                if bucket is None:
                    continue
                for kept_index in bucket:
                    if marks[kept_index] == 0:
                        marks[kept_index] = 1
                        touched_indexes.append(kept_index)
        return touched_indexes

    def _find_marked_partition_one_duplicate(
        self, phash_value: int
    ) -> Optional[int]:
        marks = self._candidate_marks
        partition_one_chunks = _partition_chunks(phash_value, 1)
        for chunk_index, chunk_value in enumerate(partition_one_chunks):
            index = self._indexes[1][chunk_index]
            for mask in self._variant_masks_by_chunk[chunk_index]:
                bucket = index.get(chunk_value ^ mask)
                if bucket is None:
                    continue
                for kept_index in bucket:
                    if marks[kept_index] != 1:
                        continue
                    if (
                        phash_value ^ self._hashes[kept_index]
                    ).bit_count() <= self._threshold:
                        return kept_index
                    marks[kept_index] = 2
        return None


def deduplicate_by_phash(
    objects: Iterable[InputT],
    threshold: int,
    *,
    sort_key: Optional[Callable[[InputT], Union[str, int]]] = None,
) -> LocalDeduplicationResult[InputT]:
    """Deduplicate local DatasetItems or item/annotation rows by pHash.

    This utility does not call the Nucleus API and does not require an API key.
    It accepts either ``DatasetItem`` objects or rows returned from
    ``items_and_annotation_generator()`` / ``items_and_annotations()`` where
    the row has an ``"item"`` key containing a ``DatasetItem``. The returned
    ``unique`` list contains the same kind of objects that were passed in.

    Args:
        objects: DatasetItems or item/annotation rows to deduplicate.
        threshold: Hamming distance threshold (0-64). Lower = stricter.
            ``0`` deduplicates exact pHash matches only.
        sort_key: Optional stable tie-break key used after pHash sorting. When
            omitted, DatasetItem.reference_id is used.

    Returns:
        LocalDeduplicationResult containing the surviving input objects and
        summary statistics.

    Raises:
        TypeError: If an input object is neither a DatasetItem nor a mapping
            with an ``"item"`` DatasetItem.
        ValueError: If threshold is invalid, or any selected DatasetItem is
            missing a 64-character binary pHash.
    """

    _validate_threshold(threshold)
    records = _normalize_records(objects, sort_key=sort_key)
    if not records:
        return LocalDeduplicationResult(
            unique=[],
            unique_dataset_items=[],
            unique_reference_ids=[],
            stats=DeduplicationStats(
                threshold=threshold,
                original_count=0,
                deduplicated_count=0,
            ),
        )

    records.sort(key=lambda record: (record.phash_value, record.stable_id))

    native_unique_records = _deduplicate_with_native(records, threshold)
    if native_unique_records is not None:
        unique_records = native_unique_records
    elif threshold == 0:
        unique_records = _deduplicate_exact(records)
    elif threshold == DEDUP_THRESHOLD_MAX:
        unique_records = [records[0]]
    elif threshold <= LOCAL_INDEX_MAX_THRESHOLD:
        unique_records = _deduplicate_with_index(records, threshold)
    else:
        unique_records = _deduplicate_with_linear_scan(records, threshold)

    return LocalDeduplicationResult(
        unique=[record.obj for record in unique_records],
        unique_dataset_items=[
            record.dataset_item for record in unique_records
        ],
        unique_reference_ids=[
            record.dataset_item.reference_id for record in unique_records
        ],
        stats=DeduplicationStats(
            threshold=threshold,
            original_count=len(records),
            deduplicated_count=len(unique_records),
        ),
    )


def _validate_threshold(threshold: int) -> None:
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold < DEDUP_THRESHOLD_MIN
        or threshold > DEDUP_THRESHOLD_MAX
    ):
        raise ValueError(
            "threshold must be an integer between 0 and 64 (inclusive)"
        )


def _normalize_records(
    objects: Iterable[InputT],
    *,
    sort_key: Optional[Callable[[InputT], Union[str, int]]],
) -> List[_DeduplicationRecord[InputT]]:
    records = []
    for ordinal, obj in enumerate(objects):
        dataset_item = _extract_dataset_item(obj)
        phash = dataset_item.phash
        if phash is None or not PHASH_REGEX.fullmatch(phash):
            ref_id = dataset_item.reference_id
            ref_msg = f" with reference_id {ref_id!r}" if ref_id else ""
            raise ValueError(
                "All DatasetItems must have a 64-character binary pHash. "
                f"Found missing or malformed pHash on item{ref_msg}."
            )

        tie_breaker = (
            sort_key(obj)
            if sort_key is not None
            else dataset_item.reference_id
        )
        stable_id = _tie_break_key(tie_breaker, ordinal)
        records.append(
            _DeduplicationRecord(
                stable_id=stable_id,
                phash_value=int(phash, 2),
                obj=obj,
                dataset_item=dataset_item,
            )
        )
    return records


def _tie_break_key(
    tie_breaker: Optional[Union[str, int]], ordinal: int
) -> _TieBreakKey:
    if tie_breaker is None:
        return (2, ordinal)
    if isinstance(tie_breaker, int):
        return (0, tie_breaker)
    return (1, tie_breaker)


def _extract_dataset_item(obj: InputT) -> DatasetItem:
    if isinstance(obj, DatasetItem):
        return obj

    if isinstance(obj, Mapping):
        item = obj.get(ITEM_KEY)
        if isinstance(item, DatasetItem):
            return item

    raise TypeError(
        "Expected each object to be a DatasetItem or a mapping with an "
        f"{ITEM_KEY!r} DatasetItem."
    )


def _deduplicate_exact(
    records: Sequence[_DeduplicationRecord[InputT]],
) -> List[_DeduplicationRecord[InputT]]:
    seen_phashes = set()
    unique_records = []
    for record in records:
        if record.phash_value in seen_phashes:
            continue
        seen_phashes.add(record.phash_value)
        unique_records.append(record)
    return unique_records


def _deduplicate_with_index(
    records: Sequence[_DeduplicationRecord[InputT]], threshold: int
) -> List[_DeduplicationRecord[InputT]]:
    index = _HammingIndex(threshold)
    unique_records = []
    for record in records:
        if index.add_if_unique(record.phash_value):
            unique_records.append(record)
    return unique_records


def _deduplicate_with_native(
    records: Sequence[_DeduplicationRecord[InputT]], threshold: int
) -> Optional[List[_DeduplicationRecord[InputT]]]:
    if _NATIVE_DEDUP is None:
        return None

    phash_values = [record.phash_value for record in records]
    kept_indexes = _NATIVE_DEDUP.deduplicate_phashes(phash_values, threshold)
    return [records[index] for index in kept_indexes]


def _deduplicate_with_linear_scan(
    records: Sequence[_DeduplicationRecord[InputT]], threshold: int
) -> List[_DeduplicationRecord[InputT]]:
    kept_hashes: List[int] = []
    unique_records = []
    for record in records:
        is_duplicate = any(
            (record.phash_value ^ kept_hash).bit_count() <= threshold
            for kept_hash in kept_hashes
        )
        if is_duplicate:
            continue
        kept_hashes.append(record.phash_value)
        unique_records.append(record)
    return unique_records


def _partition_chunks(
    phash_value: int, partition_index: int
) -> tuple[int, ...]:
    if partition_index == 1:
        phash_value = _rotate_right_64(phash_value, ROTATED_PARTITION_BITS)

    chunks = []
    shift = 64
    for chunk_bits in INDEX_CHUNK_BITS:
        shift -= chunk_bits
        chunks.append((phash_value >> shift) & ((1 << chunk_bits) - 1))
    return tuple(chunks)


def _rotate_right_64(phash_value: int, bits: int) -> int:
    return (
        (phash_value >> bits) | (phash_value << (64 - bits))
    ) & PHASH_VALUE_MASK


def _clear_candidate_marks(
    marks: bytearray, touched_indexes: List[int]
) -> None:
    for kept_index in touched_indexes:
        marks[kept_index] = 0


def _variant_masks(radius: int, chunk_bits: int) -> List[int]:
    masks = [0]
    bit_indexes = range(chunk_bits)
    for distance in range(1, radius + 1):
        for positions in itertools.combinations(bit_indexes, distance):
            mask = 0
            for position in positions:
                mask |= 1 << position
            masks.append(mask)
    return masks
