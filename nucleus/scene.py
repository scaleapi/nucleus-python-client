import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Dict, List
from nucleus.constants import (
    FRAMES_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
    POINTCLOUD_LOCATION_KEY,
)
from .annotation import is_local_path
from .dataset_item import DatasetItemType, DatasetItem


@dataclass
class Frame:
    items: Dict[str, DatasetItem] = field(default_factory=dict)
    index: Union[int, None] = None

    def __post_init__(self):
        for key, value in self.items.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, DatasetItem
            ), "All values must be DatasetItems"

    def add_item(self, item: DatasetItem, sensor_name: str):
        self.items[sensor_name] = item

    @classmethod
    def from_json(cls, payload: dict):
        items = {
            sensor: DatasetItem.from_json(item, is_scene=True)
            for sensor, item in payload.items()
        }
        return cls(items=items)

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

    def add_item(self, index: int, sensor_name: str, item: DatasetItem):
        if index not in self.frames_dict:
            new_frame = Frame(index=index, items={sensor_name: item})
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

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass
class LidarScene(Scene):
    def validate(self):
        super().validate()
        lidar_sources = flatten(
            [
                [
                    source
                    for source in frame.items.keys()
                    if frame.items[source].type == DatasetItemType.POINTCLOUD
                ]
                for frame in self.frames_dict.values()
            ]
        )
        assert (
            len(set(lidar_sources)) == 1
        ), "Each lidar scene must have exactly one lidar source"

        for frame in self.frames_dict.values():
            num_pointclouds = sum(
                [
                    int(item.type == DatasetItemType.POINTCLOUD)
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
                if is_local_path(getattr(item, POINTCLOUD_LOCATION_KEY)):
                    raise ValueError(
                        f"All paths for DatasetItems in a Scene must be remote, but {item.url} is either "
                        "local, or a remote URL type that is not supported."
                    )
