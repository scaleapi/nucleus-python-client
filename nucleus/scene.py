import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from nucleus.constants import (
    FRAMES_KEY,
    LENGTH_KEY,
    METADATA_KEY,
    NUM_SENSORS_KEY,
    REFERENCE_ID_KEY,
    POINTCLOUD_LOCATION_KEY,
    IMAGE_LOCATION_KEY,
)
from .annotation import is_local_path
from .dataset_item import DatasetItemType, DatasetItem


class Frame:
    def __init__(self, **kwargs):
        self.items = {}
        for key, value in kwargs.items():
            self.items[key] = value

    def __post_init__(self):
        for key, value in self.items.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, DatasetItem
            ), "All values must be DatasetItems"

    def __repr__(self) -> str:
        return f"Frame(items={self.items})"

    def add_item(self, item: DatasetItem, sensor_name: str):
        self.items[sensor_name] = item

    def get_item(self, sensor_name: str):
        if sensor_name not in self.items:
            raise ValueError(
                f"This frame does not have a {sensor_name} sensor"
            )
        return self.items[sensor_name]

    def get_items(self):
        return list(self.items.values())

    def get_sensors(self):
        return list(self.items.keys())

    @classmethod
    def from_json(cls, payload: dict):
        items = {
            sensor: DatasetItem.from_json(item, is_scene=True)
            for sensor, item in payload.items()
        }
        return cls(**items)

    def to_payload(self) -> dict:
        return {
            sensor: dataset_item.to_payload(is_scene=True)
            for sensor, dataset_item in self.items.items()
        }


@dataclass
class Scene(ABC):
    reference_id: str
    frames: List[Frame] = field(default_factory=list)
    metadata: Optional[dict] = None

    def __post_init__(self):
        self.sensors = set(
            flatten([frame.get_sensors() for frame in self.frames])
        )
        self.frames_dict = dict(enumerate(self.frames))

    @property
    def length(self) -> int:
        return len(self.frames_dict)

    @property
    def num_sensors(self) -> int:
        return len(self.get_sensors())

    def validate(self):
        assert self.length > 0, "Must have at least 1 frame in a scene"
        for frame in self.frames_dict.values():
            assert isinstance(
                frame, Frame
            ), "Each frame in a scene must be a Frame object"

    def add_item(self, index: int, sensor_name: str, item: DatasetItem):
        self.sensors.add(sensor_name)
        if index not in self.frames_dict:
            new_frame = Frame(**{sensor_name: item})
            self.frames_dict[index] = new_frame
        else:
            self.frames_dict[index].items[sensor_name] = item

    def add_frame(self, frame: Frame, index: int, update: bool = False):
        if (
            index not in self.frames_dict
            or index in self.frames_dict
            and update
        ):
            self.frames_dict[index] = frame
            self.sensors.update(frame.get_sensors())

    def get_frame(self, index: int):
        if index not in self.frames_dict:
            raise ValueError(
                f"This scene does not have a frame at index {index}"
            )
        return self.frames_dict[index]

    def get_frames(self):
        return [
            frame
            for _, frame in sorted(
                self.frames_dict.items(), key=lambda x: x[0]
            )
        ]

    def get_sensors(self):
        return list(self.sensors)

    def get_item(self, index: int, sensor_name: str):
        frame = self.get_frame(index)
        return frame.get_item(sensor_name)

    def get_items_from_sensor(self, sensor_name: str):
        if sensor_name not in self.sensors:
            raise ValueError(
                f"This scene does not have a {sensor_name} sensor"
            )
        items_from_sensor = []
        for frame in self.frames_dict.values():
            try:
                sensor_item = frame.get_item(sensor_name)
                items_from_sensor.append(sensor_item)
            except ValueError:
                # This sensor is not present at current frame
                items_from_sensor.append(None)
        return items_from_sensor

    def get_items(self):
        return flatten([frame.get_items() for frame in self.get_frames()])

    def info(self):
        return {
            REFERENCE_ID_KEY: self.reference_id,
            LENGTH_KEY: self.length,
            NUM_SENSORS_KEY: self.num_sensors,
        }

    def validate_frames_dict(self):
        is_continuous = set(list(range(len(self.frames_dict)))) == set(
            self.frames_dict.keys()
        )
        assert (
            is_continuous
        ), "frames must be 0-indexed and continuous (no missing frames)"

    @classmethod
    def from_json(cls, payload: dict):
        frames_payload = payload.get(FRAMES_KEY, [])
        frames = [Frame.from_json(frame) for frame in frames_payload]
        return cls(
            reference_id=payload[REFERENCE_ID_KEY],
            frames=frames,
            metadata=payload.get(METADATA_KEY, None),
        )

    def to_payload(self) -> dict:
        self.validate_frames_dict()
        ordered_frames = self.get_frames()
        frames_payload = [frame.to_payload() for frame in ordered_frames]
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
            FRAMES_KEY: frames_payload,
        }
        if self.metadata:
            payload[METADATA_KEY] = self.metadata
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass
class LidarScene(Scene):
    def __repr__(self) -> str:
        return f"LidarScene(reference_id='{self.reference_id}', frames={self.get_frames()}, metadata={self.metadata})"

    def validate(self):
        super().validate()
        lidar_sensors = flatten(
            [
                [
                    sensor
                    for sensor in frame.items.keys()
                    if frame.items[sensor].type == DatasetItemType.POINTCLOUD
                ]
                for frame in self.frames_dict.values()
            ]
        )
        assert (
            len(set(lidar_sensors)) == 1
        ), "Each lidar scene must have exactly one lidar sensor"

        for frame in self.frames_dict.values():
            num_pointclouds = sum(
                [
                    int(item.type == DatasetItemType.POINTCLOUD)
                    for item in frame.get_items()
                ]
            )
            assert (
                num_pointclouds == 1
            ), "Each frame of a lidar scene must have exactly 1 pointcloud"


def flatten(t):
    return [item for sublist in t for item in sublist]


def check_all_scene_paths_remote(scenes: List[LidarScene]):
    for scene in scenes:
        for item in scene.get_items():
            pointcloud_location = getattr(item, POINTCLOUD_LOCATION_KEY)
            if pointcloud_location and is_local_path(pointcloud_location):
                raise ValueError(
                    f"All paths for DatasetItems in a Scene must be remote, but {item.pointcloud_location} is either "
                    "local, or a remote URL type that is not supported."
                )
            image_location = getattr(item, IMAGE_LOCATION_KEY)
            if image_location and is_local_path(image_location):
                raise ValueError(
                    f"All paths for DatasetItems in a Scene must be remote, but {item.image_location} is either "
                    "local, or a remote URL type that is not supported."
                )
