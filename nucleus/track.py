import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

from .constants import (
    DATASET_ID_KEY,
    METADATA_KEY,
    OVERWRITE_KEY,
    REFERENCE_ID_KEY,
)

if TYPE_CHECKING:
    from . import Connection


@dataclass  # pylint: disable=R0902
class Track:  # pylint: disable=R0902
    """A track is a class of objects (annotation or prediction) that forms a one-to-many relationship
    with objects, wherein an object is an instance of a track.

    Args:
        reference_id (str): A user-specified name of the track that describes the class of objects it represents.
        metadata: Arbitrary key/value dictionary of info to attach to this track.
    """

    _connection: "Connection"
    dataset_id: str
    reference_id: str
    metadata: Optional[dict] = None

    def __repr__(self):
        return f"Track(dataset_id='{self.dataset_id}', reference_id='{self.reference_id}', metadata={self.metadata})"

    def __eq__(self, other):
        return (
            (self.dataset_id == other.dataset_id)
            and (self.reference_id == other.reference_id)
            and (self.metadata == other.metadata)
        )

    @classmethod
    def from_json(cls, payload: dict, connection: "Connection"):
        """Instantiates track object from schematized JSON dict payload."""
        return cls(
            _connection=connection,
            reference_id=str(payload[REFERENCE_ID_KEY]),
            dataset_id=str(payload[DATASET_ID_KEY]),
            metadata=payload.get(METADATA_KEY, None),
        )

    def to_payload(self) -> dict:
        """Serializes track object to schematized JSON dict."""
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
            DATASET_ID_KEY: self.dataset_id,
            METADATA_KEY: self.metadata,
        }

        return payload

    def to_json(self) -> str:
        """Serializes track object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)

    def update(
        self,
        metadata: Optional[dict] = None,
        overwrite_metadata: bool = False,
    ) -> None:
        """
        Updates the Track's metadata.

        Parameters:
            metadata (Optional[dict]): An arbitrary dictionary of additional data about this track that can be stored
                and retrieved.
            overwrite_metadata (Optional[bool]): If metadata is provided and overwrite_metadata = True, then the track's
                entire metadata object will be overwritten. Otherwise, only the keys in metadata will be overwritten.
        """

        self._connection.make_request(
            payload={
                REFERENCE_ID_KEY: self.reference_id,
                METADATA_KEY: metadata,
                OVERWRITE_KEY: overwrite_metadata,
            },
            route=f"dataset/{self.dataset_id}/tracks",
            requests_command=requests.post,
        )
        self.metadata = (
            metadata
            if overwrite_metadata
            else (
                {**self.metadata, **metadata}
                if self.metadata is not None and metadata is not None
                else metadata
            )
        )
