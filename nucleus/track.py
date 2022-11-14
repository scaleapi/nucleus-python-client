import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests

from .constants import (
    DATASET_ID_KEY,
    METADATA_KEY,
    OVERWRITE_KEY,
    REFERENCE_ID_KEY,
    SCENE_REFERENCE_ID_KEY,
)

if TYPE_CHECKING:
    from . import NucleusClient


@dataclass  # pylint: disable=R0902
class Track:  # pylint: disable=R0902
    """A track is a class of objects (annotation or prediction) that forms a one-to-many relationship
    with objects, wherein an object is an instance of a track.

    Args:
        reference_id (str): A user-specified name of the track that describes the class of objects it represents.
        scene_reference_id (Optional[str]): A user-specified reference ID for the scene this track belongs to.
        metadata: Arbitrary key/value dictionary of info to attach to this track.
    """

    _client: "NucleusClient"
    reference_id: str
    dataset_id: str
    scene_reference_id: Optional[str] = None
    metadata: Optional[dict] = None

    @classmethod
    def from_json(cls, payload: dict, client: "NucleusClient"):
        """Instantiates track object from schematized JSON dict payload."""
        return cls(
            _client=client,
            reference_id=str(payload[REFERENCE_ID_KEY]),
            dataset_id=str(payload[DATASET_ID_KEY]),
            scene_reference_id=payload.get(SCENE_REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, None),
        )

    def to_payload(self) -> dict:
        """Serializes track object to schematized JSON dict."""
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
            DATASET_ID_KEY: self.dataset_id,
            SCENE_REFERENCE_ID_KEY: self.scene_reference_id,
            METADATA_KEY: self.metadata,
        }

        return payload

    def to_json(self) -> str:
        """Serializes track object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)

    def update(
        self,
        scene_reference_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        overwrite_metadata: bool = False,
    ) -> None:
        """
        Updates the Track's scene_reference_id or metadata.

        Parameters:
            scene_reference_id (Optional[str]): The reference ID of the scene this track links to.
            metadata (Optional[dict]): An arbitrary dictionary of additional data about this track that can be stored
                and retrieved.
            overwrite_metadata (Optional[bool]): If metadata is provided and overwrite_metadata = True, then the track's
                entire metadata object will be overwritten. Otherwise, only the keys in metadata will be overwritten.
        """

        assert (
            scene_reference_id is not None and metadata is not None
        ), "Must provide scene_reference_id or metadata"

        self._client.make_request(
            payload={
                REFERENCE_ID_KEY: self.reference_id,
                SCENE_REFERENCE_ID_KEY: scene_reference_id,
                METADATA_KEY: metadata,
                OVERWRITE_KEY: overwrite_metadata,
            },
            route=f"dataset/{self.dataset_id}/track/update",
            requests_command=requests.post,
        )
        self.scene_reference_id = (
            scene_reference_id
            if scene_reference_id
            else self.scene_reference_id
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
