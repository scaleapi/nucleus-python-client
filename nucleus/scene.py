import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Set
from enum import Enum
from nucleus.constants import (
    CAMERA_PARAMS_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    URL_KEY,
)
from .annotation import Point3D
from .utils import flatten


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"


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

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            url=payload.get(URL_KEY, ""),
            type=payload.get(TYPE_KEY, ""),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, None),
            camera_params=payload.get(CAMERA_PARAMS_KEY, None),
        )

    def to_payload(self) -> dict:
        return {
            URL_KEY: self.url,
            TYPE_KEY: self.type,
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
            CAMERA_PARAMS_KEY: self.camera_params,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass
class Frame:
    items: Dict[str, SceneDatasetItem]

    def __post_init__(self):
        for key, value in self.items.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, SceneDatasetItem
            ), "All values must be SceneDatasetItems"

    def add_item(self, item: SceneDatasetItem, sensor_name: str):
        self.items[sensor_name] = item


@dataclass
class Scene:
    frames: List[Frame]
    reference_id: str
    metadata: Optional[dict] = None

    def __post_init__(self):
        assert isinstance(self.frames, List), "frames must be a list"
        for frame in self.frames:
            assert isinstance(
                frame, Frame
            ), "each element of frames must be a Frame object"
        assert len(self.frames) > 0, "frames must have length of at least 1"
        assert isinstance(
            self.reference_id, str
        ), "reference_id must be a string"


@dataclass
class LidarScene(Scene):
    def validate(self):
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

        for frame in self.frames:
            num_pointclouds = sum(
                [
                    int(item.type == DatasetItemType.POINTCLOUD)
                    for item in frame.values()
                ]
            )
            assert (
                num_pointclouds == 1
            ), "Each frame of a lidar scene must have exactly 1 pointcloud"
