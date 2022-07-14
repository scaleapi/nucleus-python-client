from typing import Dict

import numpy as np
from PIL import Image

try:
    import fsspec
except (ModuleNotFoundError, OSError):
    from ..package_not_installed import PackageNotInstalled

    fsspec = PackageNotInstalled


class SegmentationMaskLoader:
    def __init__(self, fs: fsspec):
        self.fs = fs

    def fetch(self, url: str):
        with self.fs.open(url) as fh:
            img = Image.open(fh)
        return np.asarray(img)


class InMemoryLoader:
    """We use this loader in the tests, this allows us to serve images from memory instead of fetching
    from a filesystem.
    """

    def __init__(self, url_to_array: Dict[str, np.ndarray]):
        self.url_to_array = url_to_array

    def fetch(self, url: str):
        array = self.url_to_array[url]
        return array
