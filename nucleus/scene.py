from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"
    VIDEO = "video"


@dataclass
class SceneDatasetItem:
    url: str
    type: DatasetItemType
    reference_id: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class Frame:
    items: Dict[str, SceneDatasetItem]

    def __post_init__(self):
        for key, value in self.items.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, SceneDatasetItem
            ), "All values must be SceneDatasetItems"


@dataclass
class Scene:
    frames: List[Frame]
    reference_id: str
    metadata: Optional[dict] = None


@dataclass
class LidarScene(Scene):
    def __post_init__(self):
        # do validation here for lidar scene
        pass
