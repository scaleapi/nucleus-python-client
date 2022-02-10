from enum import Enum
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from . import NucleusClient


# Wording set to match with backend enum
class ExportMetadataType(Enum):
    SCENES = "scene"
    DATASET_ITEMS = "item"


class MetadataManager:
    """
    Helper class for managing metadata updates on a scene or dataset item.
    Do not call directly, use the dataset class methods: `update_scene_metadata` or `update_item_metadata`
    """

    def __init__(
        self,
        dataset_id: str,
        client: "NucleusClient",
        raw_mappings: Dict[str, dict],
        level: ExportMetadataType,
    ):
        self.dataset_id = dataset_id
        self._client = client
        self.raw_mappings = raw_mappings
        self.level = level

        self._payload = self._format_mappings()

    def _format_mappings(self):
        payload = []
        for ref_id, meta in self.raw_mappings.items():
            payload.append({"reference_id": ref_id, "metadata": meta})
        return payload

    def update(self):
        payload = {"metadata": self._payload, "level": self.level.value}
        resp = self._client.make_request(
            payload=payload, route=f"dataset/{self.dataset_id}/metadata"
        )
        return resp
