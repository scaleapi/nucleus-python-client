from dataclasses import dataclass
from typing import List


@dataclass
class DeduplicationStats:
    threshold: int
    original_count: int
    deduplicated_count: int


@dataclass
class DeduplicationResult:
    unique_item_ids: List[str]  # Internal dataset item IDs
    unique_reference_ids: List[str]  # User-defined reference IDs
    stats: DeduplicationStats
