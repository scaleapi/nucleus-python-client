import json
import os.path
from dataclasses import dataclass
from typing import List, Optional

from .constants import (
    DATASET_ITEM_ID_KEY,
    IMAGE_URL_KEY,
    METADATA_KEY,
    ORIGINAL_IMAGE_URL_KEY,
    REFERENCE_ID_KEY,
)


@dataclass
class DatasetItem:

    image_location: str
    reference_id: Optional[str] = None
    item_id: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        self.local = is_local_path(self.image_location)

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

    def local_file_exists(self):
        return os.path.isfile(self.image_location)

    def to_payload(self) -> dict:
        payload = {
            IMAGE_URL_KEY: self.image_location,
            METADATA_KEY: self.metadata or {},
        }
        if self.reference_id:
            payload[REFERENCE_ID_KEY] = self.reference_id
        if self.item_id:
            payload[DATASET_ITEM_ID_KEY] = self.item_id
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_payload())


def is_local_path(path: str) -> bool:
    path_components = [comp.lower() for comp in path.split("/")]
    return path_components[0] not in {"https:", "http:", "s3:", "gs:"}


def check_all_paths_remote(dataset_items: List[DatasetItem]):
    for item in dataset_items:
        if is_local_path(item.image_location):
            raise ValueError(
                f"All paths must be remote, but {item.image_location} is either "
                "local, or a remote URL type that is not supported."
            )
