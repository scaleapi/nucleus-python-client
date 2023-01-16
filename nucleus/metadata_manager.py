from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

from .async_job import AsyncJob
from .camera_params import CameraParams
from .constants import CAMERA_PARAMS_KEY

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
        asynchronous: bool,
    ):
        self.dataset_id = dataset_id
        self._client = client
        self.raw_mappings = raw_mappings
        self.level = level
        self.asynchronous = asynchronous

        if len(self.raw_mappings) > 500 and not self.asynchronous:
            raise Exception(
                "Number of items to update is too large to perform it synchronously. "
                "Consider running the metadata_update with `asynchronous=True`, to avoid timeouts."
            )

        self._payload = self._format_mappings()

    def _extract_camera_params(self, metadata: dict) -> Optional[CameraParams]:
        camera_params = metadata.get(CAMERA_PARAMS_KEY, None)
        if camera_params is None:
            return None
        return CameraParams.from_json(camera_params)

    def _format_mappings(self):
        payloads = []
        for ref_id, meta in self.raw_mappings.items():
            payload = {"reference_id": ref_id, "metadata": meta}

            if self.level.value == ExportMetadataType.DATASET_ITEMS.value:
                camera_params = self._extract_camera_params(meta)
                if camera_params:
                    payload[CAMERA_PARAMS_KEY] = camera_params.to_payload()

            payloads.append(payload)
        return payloads

    def update(self):
        payload = {"metadata": self._payload, "level": self.level.value}
        is_async = int(self.asynchronous)
        try:
            resp = self._client.make_request(
                payload=payload,
                route=f"dataset/{self.dataset_id}/metadata?async={is_async}",
            )
            if self.asynchronous:
                return AsyncJob.from_json(resp, self._client)
            return resp
        except Exception as e:  # pylint: disable=W0703
            print(
                "Failed to complete the request. If a timeout occurred, consider running the "
                "metadata_update with `asynchronous=True`."
            )
            print(f"Request failed with:\n\n{e}")
            return None
