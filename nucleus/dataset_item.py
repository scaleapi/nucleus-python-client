import os.path
from .errors import FileNotFoundError
from .constants import (
    IMAGE_URL_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
)


class DatasetItem:
    def __init__(
        self,
        image_location: str,
        reference_id: str = None,
        item_id: str = None,
        metadata: dict = None,
    ):

        self.image_url = image_location
        self.local = self._is_local_path(image_location)
        self.reference_id = reference_id
        self.metadata = metadata

        if self.local and not self._local_file_exists(image_location):
            raise FileNotFoundError()

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            image_location=payload.get(IMAGE_URL_KEY, ""),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __repr__(self):
        return str(self.to_payload())

    def _is_local_path(self, path: str) -> bool:
        path_components = [comp.lower() for comp in path.split("/")]
        return not (
            "https:" in path_components
            or "http:" in path_components
            or "s3:" in path_components
        )

    def _local_file_exists(self, path: str):
        return os.path.isfile(path)

    def to_payload(self) -> dict:
        payload = {IMAGE_URL_KEY: self.image_url, METADATA_KEY: self.metadata}
        if self.reference_id:
            payload[REFERENCE_ID_KEY] = self.reference_id
        return payload
