import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Dict, List, Set
from enum import Enum
from nucleus.constants import (
    CAMERA_PARAMS_KEY,
    CX_KEY,
    CY_KEY,
    FRAMES_KEY,
    FX_KEY,
    FY_KEY,
    HEADING_KEY,
    INDEX_KEY,
    ITEMS_KEY,
    METADATA_KEY,
    POSITION_KEY,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    URL_KEY,
    W_KEY,
    X_KEY,
    Y_KEY,
    Z_KEY,
)
from .annotation import Point3D
from .dataset_item import is_local_path


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        return cls(
            payload[X_KEY], payload[Y_KEY], payload[Z_KEY], payload[W_KEY]
        )

    def to_payload(self) -> dict:
        return {
            X_KEY: self.x,
            Y_KEY: self.y,
            Z_KEY: self.z,
            W_KEY: self.w,
        }


@dataclass
class CameraParams:
    position: Point3D
    heading: Quaternion
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_json(cls, payload: Dict[str, Any]):
        return cls(
            Point3D.from_json(payload[POSITION_KEY]),
            Quaternion.from_json(payload[HEADING_KEY]),
            payload[FX_KEY],
            payload[FY_KEY],
            payload[CX_KEY],
            payload[CY_KEY],
        )

    def to_payload(self) -> dict:
        return {
            POSITION_KEY: self.position.to_payload(),
            HEADING_KEY: self.heading.to_payload(),
            FX_KEY: self.fx,
            FY_KEY: self.fy,
            CX_KEY: self.cx,
            CY_KEY: self.cy,
        }


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
        camera_params = (
            CameraParams.from_json(payload[CAMERA_PARAMS_KEY])
            if payload.get(CAMERA_PARAMS_KEY, None)
            else None
        )
        return cls(
            url=payload[URL_KEY],
            type=payload[TYPE_KEY],
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, None),
            camera_params=camera_params,
        )

    def to_payload(self) -> dict:
        payload: Dict[str, Any] = {
            URL_KEY: self.url,
            TYPE_KEY: self.type,
        }
        if self.reference_id:
            payload[REFERENCE_ID_KEY] = self.reference_id
        if self.metadata:
            payload[METADATA_KEY] = self.metadata
        if self.camera_params:
            payload[CAMERA_PARAMS_KEY] = self.camera_params.to_payload()
        return payload

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
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
            FRAMES_KEY: frames_payload,
        }
        if self.metadata:
            payload[METADATA_KEY] = self.metadata
        return payload


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
                    int(item.type == DatasetItemType.POINTCLOUD.value)
                    for item in frame.items.values()
                ]
            )
            assert (
                num_pointclouds == 1
            ), "Each frame of a lidar scene must have exactly 1 pointcloud"


def flatten(t):
    return [item for sublist in t for item in sublist]


def check_all_scene_paths_remote(scenes: List[LidarScene]):
    for scene in scenes:
        for frame in scene.frames_dict.values():
            for item in frame.items.values():
                if is_local_path(getattr(item, URL_KEY)):
                    raise ValueError(
                        f"All paths for SceneDatasetItems must be remote, but {item.url} is either "
                        "local, or a remote URL type that is not supported."
                    )
