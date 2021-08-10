import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Set
from enum import Enum
from nucleus.constants import (
    CAMERA_PARAMS_KEY,
    FRAMES_KEY,
    INDEX_KEY,
    ITEMS_KEY,
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
    type: str
    reference_id: Optional[str] = None
    metadata: Optional[dict] = None
    camera_params: Optional[CameraParams] = None

    def __post_init__(self):
        assert self.type in [
            e.value for e in DatasetItemType
        ], "type must be one of DatasetItemType's enum values i.e. `image` or `pointcloud`"

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
    index: Union[int, None] = None
    items: Dict[str, SceneDatasetItem] = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.items.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, SceneDatasetItem
            ), "All values must be SceneDatasetItems"

    def add_item(self, item: SceneDatasetItem, sensor_name: str):
        self.items[sensor_name] = item

    def to_payload(self) -> dict:
        return {
            INDEX_KEY: self.index,
            ITEMS_KEY: {
                sensor: scene_dataset_item.to_payload()
                for sensor, scene_dataset_item in self.items.items()
            },
        }


@dataclass
class Scene(ABC):
    reference_id: str
    frames: List[Frame] = field(default_factory=list)
    metadata: Optional[dict] = None

    def __post_init__(self):
        self.check_valid_frame_indices()
        if all((frame.index is not None for frame in self.frames)):
            self.frames_dict = {frame.index: frame for frame in self.frames}
        else:
            indexed_frames = [
                Frame(index=i, items=frame.items)
                for i, frame in enumerate(self.frames)
            ]
            self.frames_dict = dict(enumerate(indexed_frames))

    def check_valid_frame_indices(self):
        infer_from_list_position = all(
            (frame.index is None for frame in self.frames)
        )
        explicit_frame_order = all(
            (frame.index is not None for frame in self.frames)
        )
        assert (
            infer_from_list_position or explicit_frame_order
        ), "Must specify index explicitly for all frames or infer from list position for all frames"

    def validate(self):
        assert (
            len(self.frames_dict) > 0
        ), "Must have at least 1 frame in a scene"
        for frame in self.frames_dict.values():
            assert isinstance(
                frame, Frame
            ), "Each frame in a scene must be a Frame object"

    def add_item(self, index: int, sensor_name: str, item: SceneDatasetItem):
        if index not in self.frames_dict:
            new_frame = Frame(index, {sensor_name: item})
            self.frames_dict[index] = new_frame
        else:
            self.frames_dict[index].items[sensor_name] = item

    def add_frame(self, frame: Frame, update: bool = False):
        assert (
            frame.index is not None
        ), "Must specify index explicitly when calling add_frame"
        if (
            frame.index not in self.frames_dict
            or frame.index in self.frames_dict
            and update
        ):
            self.frames_dict[frame.index] = frame

    def to_payload(self) -> dict:
        ordered_frames = [
            frame
            for _, frame in sorted(
                self.frames_dict.items(), key=lambda x: x[0]
            )
        ]
        frames_payload = [frame.to_payload() for frame in ordered_frames]
        return {
            REFERENCE_ID_KEY: self.reference_id,
            FRAMES_KEY: frames_payload,
            METADATA_KEY: self.metadata,
        }


@dataclass
class LidarScene(Scene):
    # TODO: call validate in scene upload
    def validate(self):
        super().validate()
        lidar_sources = flatten(
            [
                [
                    source
                    for source in frame.items.keys()
                    if frame.items[source].type
                    == DatasetItemType.POINTCLOUD.value
                ]
                for frame in self.frames_dict.values()
            ]
        )
        assert (
            len(Set(lidar_sources)) == 1
        ), "Each lidar scene must have exactly one lidar source"

        for frame in self.frames_dict.values():
            num_pointclouds = sum(
                [
                    int(item.type == DatasetItemType.POINTCLOUD)
                    for item in frame.values()
                ]
            )
            assert (
                num_pointclouds == 1
            ), "Each frame of a lidar scene must have exactly 1 pointcloud"
