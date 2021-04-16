from dataclasses import dataclass
import os.path
from typing import Optional
from .constants import (
    IMAGE_URL_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
    ORIGINAL_IMAGE_URL_KEY,
    DATASET_ITEM_ID_KEY,
)


@dataclass
class DatasetItem:

    image_location: str
    reference_id: Optional[str] = None
    item_id: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        self.image_url = self.image_location
        self.local = self._is_local_path(self.image_location)

    @classmethod
    def from_json(cls, payload: dict):
        url = payload.get(IMAGE_URL_KEY, "") or payload.get(
            ORIGINAL_IMAGE_URL_KEY, ""
        )
        return cls(
            image_location=url,
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def _is_local_path(self, path: str) -> bool:
        path_components = [comp.lower() for comp in path.split("/")]
        return not (
            "https:" in path_components
            or "http:" in path_components
            or "s3:" in path_components
        )

    def local_file_exists(self):
        return os.path.isfile(self.image_url)

    def to_payload(self) -> dict:
        payload = {
            IMAGE_URL_KEY: self.image_url,
            METADATA_KEY: self.metadata or {},
        }
        if self.reference_id:
            payload[REFERENCE_ID_KEY] = self.reference_id
        if self.item_id:
            payload[DATASET_ITEM_ID_KEY] = self.item_id
        return payload
