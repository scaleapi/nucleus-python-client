import abc
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import numpy as np


class SegmentationMaskLoader(abc.ABC):
    @abc.abstractmethod
    def fetch(self, url: str) -> "np.ndarray":
        pass


class DummyLoader(SegmentationMaskLoader):
    def fetch(self, url: str) -> "np.ndarray":
        raise NotImplementedError(
            "This dummy loader has to be replaced with an actual implementation of an image loader"
        )


class InMemoryLoader(SegmentationMaskLoader):
    """We use this loader in the tests, this allows us to serve images from memory instead of fetching
    from a filesystem.
    """

    def __init__(self, url_to_array: Dict[str, "np.ndarray"]):
        self.url_to_array = url_to_array
        super().__init__()

    def fetch(self, url: str):
        array = self.url_to_array[url]
        return array
