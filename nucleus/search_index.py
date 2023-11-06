from dataclasses import dataclass
from enum import Enum

from nucleus.constants import (
    EMBEDDING_TYPE_KEY,
    ID_KEY,
    INDEX_LEVEL_KEY,
    INDEX_TYPE_KEY,
    STATUS_KEY,
)


class IndexType(str, Enum):
    INTERNAL = "Internal"
    CUSTOM = "Custom"


class IndexLevel(str, Enum):
    IMAGE = "Image"
    OBJECT = "Object"


class IndexStatus(str, Enum):
    STARTED = "Started"
    COMPLETED = "Completed"
    ERRORED = "Errored"


@dataclass
class SearchIndex:
    """Represents a Search Index belonging to a Dataset.

    Search Indexes contain generated embeddings for each item in the dataset,
    and are used by the Autotag and the Similarity Search functionality.
    """

    id: str
    status: IndexStatus
    index_type: IndexType
    index_level: IndexLevel
    embedding_type: str

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            id=payload[ID_KEY],
            status=payload[STATUS_KEY],
            index_type=payload[INDEX_TYPE_KEY],
            index_level=payload[INDEX_LEVEL_KEY],
            embedding_type=payload[EMBEDDING_TYPE_KEY],
        )
