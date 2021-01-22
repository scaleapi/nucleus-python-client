import os.path
from .errors import FileNotFoundError

class DatasetItem:

    def __init__(self, image_location: str, reference_id: str, metadata: dict):
        
        self.image_url = image_location
        self.local = self._is_local_path(image_location)
        self.reference_id = reference_id
        self.metadata = metadata

        if self.local and not self._local_file_exists(image_location):
            raise FileNotFoundError()

    def _is_local_path(self, path: str) -> bool:
        path_components = path.split('/')
        return not ('https:' in path_components  or 'http:' in path_components or 's3:' in path_components)

    def _local_file_exists(self, path):
        return os.path.isfile(path)

    def to_payload(self) -> dict:
        return {
            "image_url": self.image_url,
            "reference_id": self.reference_id,
            "metadata": self.metadata
        }