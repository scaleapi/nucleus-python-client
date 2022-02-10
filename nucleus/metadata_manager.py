from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, TYPE_CHECKING, Union


from nucleus.pydantic_base import DictCompatibleModel

if TYPE_CHECKING:
    from . import NucleusClient


# Wording set to match with backend enum
class ExportMetadataType(Enum):
    SCENES = "scene"
    DATASET_ITEMS = "item"


class DatasetItemMetadata(DictCompatibleModel):
    reference_id: str
    metadata: dict

    def to_dict(self):
        return {"reference_id": self.reference_id, "metadata": self.metadata}

@dataclass
class MetadataPayload:
    reference_id: str
    metadata: dict

    def to_dict(self):
        return {
            "reference_id": self.reference_id,
            "metadata": self.metadata,
        }

class MetadataManager:
    """
    Helper class for managing metadata updates on a scene or dataset item.
    Please note, only updating of existing keys, or adding new keys is allowed at the moment.
    It does not support metadata deletion.
    """
    def __init__(self, dataset_id: str, client: "NucleusClient", raw_mappings: dict[str, dict], level: ExportMetadataType ):
        self.dataset_id = dataset_id
        self._client = client
        self.raw_mappings = raw_mappings
        self.level = level

        self._payload = self._format_mappings()

    def _format_mappings(self):
        payload = []
        for ref_id, meta in self.raw_mappings.items():
            payload.append({"reference_id": ref_id, "metadata":meta})
        return payload

    def update(self):
        payload = {"metadata": self._payload, "level": self.level.value}
        resp = self._client.make_request(
            payload=payload, route=f"dataset/{self.dataset_id}/metadata"
        )
        return resp
