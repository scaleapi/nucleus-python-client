from dataclasses import dataclass
from typing import Optional, Dict, List, Set
from enum import Enum
from .annotation import Point3D
from .utils import flatten


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"
    VIDEO = "video"


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class CameraParams:
    position: Point3D
    heading: Quaternion
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class SceneDatasetItem:
    url: str
    type: DatasetItemType
    reference_id: Optional[str] = None
    metadata: Optional[dict] = None
    camera_params: Optional[CameraParams] = None


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
        lidar_sources = flatten(
            [
                [
                    source
                    for source in frame.items.keys()
                    if frame.items[source].type == DatasetItemType.POINTCLOUD
                ]
                for frame in self.frames
            ]
        )
        assert (
            len(Set(lidar_sources)) == 1
        ), "Each lidar scene must have exactly one lidar source"

        # TODO: check single pointcloud per frame
