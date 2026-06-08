from collections.abc import Iterable

def deduplicate_phashes(
    phashes: Iterable[int], threshold: int
) -> list[int]: ...
