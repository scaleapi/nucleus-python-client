import json
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from nucleus.constants import (
    FRAME_RATE_KEY,
    FRAMES_KEY,
    IMAGE_LOCATION_KEY,
    LENGTH_KEY,
    METADATA_KEY,
    NUM_SENSORS_KEY,
    POINTCLOUD_LOCATION_KEY,
    REFERENCE_ID_KEY,
    UPLOAD_TO_SCALE_KEY,
    VIDEO_LOCATION_KEY,
    VIDEO_URL_KEY,
)

from .annotation import is_local_path
from .dataset_item import (
    DatasetItem,
    DatasetItemType,
    check_for_duplicate_reference_ids,
)


class Frame:
    """Collection of sensor data pertaining to a single time step.

    For 3D data, each Frame houses a sensor-to-data mapping and must have exactly
    one pointcloud with any number of camera images.

    Parameters:
        **kwargs (Dict[str, :class:`DatasetItem`]): Mappings from sensor name
          to dataset item. Each frame of a lidar scene must contain exactly one
          pointcloud and any number of images (e.g. from different angles).

    Refer to our `guide to uploading 3D data
    <https://docs.nucleus.scale.com/docs/uploading-3d-data>`_ for more info!
    """

    def __init__(self, **kwargs):
        self.items: Dict[str, DatasetItem] = {}
        for key, value in kwargs.items():
            assert isinstance(key, str), "All keys must be names of sensors"
            assert isinstance(
                value, DatasetItem
            ), f"All values must be DatasetItems, instead got type {type(value)}"
            self.items[key] = value

        check_for_duplicate_reference_ids(list(self.items.values()))

    def __repr__(self) -> str:
        return f"Frame(items={self.items})"

    def __eq__(self, other):
        for key, value in self.items.items():
            if key not in other.items:
                return False
            if value != other.items[key]:
                return False
        return True

    def add_item(self, item: DatasetItem, sensor_name: str) -> None:
        """Adds DatasetItem object to frame as sensor data.

        Parameters:
            item (:class:`DatasetItem`): Pointcloud or camera image item to add.
            sensor_name: Name of the sensor, e.g. "lidar" or "front_cam."
        """
        self.items[sensor_name] = item

    def get_item(self, sensor_name: str) -> DatasetItem:
        """Fetches the DatasetItem object associated with the given sensor.

        Parameters:
            sensor_name: Name of the sensor, e.g. "lidar" or "front_cam."

        Returns:
            :class:`DatasetItem`: DatasetItem object pertaining to the sensor.
        """
        if sensor_name not in self.items:
            raise ValueError(
                f"This frame does not have a {sensor_name} sensor"
            )
        return self.items[sensor_name]

    def get_items(self) -> List[DatasetItem]:
        """Fetches all items in the frame.

        Returns:
            List[:class:`DatasetItem`]: List of all DatasetItem objects in the frame.
        """
        return list(self.items.values())

    def get_sensors(self) -> List[str]:
        """Fetches all sensor names of the frame.

        Returns:
            List of all sensor names of the frame."""
        return list(self.items.keys())

    @classmethod
    def from_json(cls, payload: dict):
        """Instantiates frame object from schematized JSON dict payload."""
        items = {
            sensor: DatasetItem.from_json(item)
            for sensor, item in payload.items()
        }
        return cls(**items)

    def to_payload(self) -> dict:
        """Serializes frame object to schematized JSON dict."""
        return {
            sensor: dataset_item.to_payload(is_scene=True)
            for sensor, dataset_item in self.items.items()
        }


@dataclass
class Scene(ABC):
    reference_id: str
    frames: List[Frame] = field(default_factory=list)
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        self.sensors = set(
            flatten([frame.get_sensors() for frame in self.frames])
        )
        self.frames_dict = dict(enumerate(self.frames))
        if self.metadata is None:
            self.metadata = {}

        self.validate()

    def __eq__(self, other):
        return all(
            [
                self.reference_id == other.reference_id,
                self.frames == other.frames,
                self.metadata == other.metadata,
            ]
        )

    @property
    def length(self) -> int:
        """Number of frames in the scene."""
        return len(self.frames_dict)

    @property
    def num_sensors(self) -> int:
        """Number of sensors in the scene."""
        return len(self.get_sensors())

    def validate(self):
        # TODO: make private
        assert self.length > 0, "Must have at least 1 frame in a scene"
        all_items = []
        for frame in self.frames_dict.values():
            assert isinstance(
                frame, Frame
            ), "Each frame in a scene must be a Frame object"
            all_items.extend(frame.get_items())

        check_for_duplicate_reference_ids(all_items)

    def add_item(
        self, index: int, sensor_name: str, item: DatasetItem
    ) -> None:
        """Adds DatasetItem to the specified frame as sensor data.

        Parameters:
            index: Serial index of the frame to which to add the item.
            item (:class:`DatasetItem`): Pointcloud or camera image item to add.
            sensor_name: Name of the sensor, e.g. "lidar" or "front_cam."
        """
        self.sensors.add(sensor_name)
        if index not in self.frames_dict:
            new_frame = Frame(**{sensor_name: item})
            self.frames_dict[index] = new_frame
        else:
            self.frames_dict[index].items[sensor_name] = item

    def add_frame(
        self, frame: Frame, index: int, update: bool = False
    ) -> None:
        """Adds frame to scene at the specified index.

        Parameters:
            frame (:class:`Frame`): Frame object to add.
            index: Serial index at which to add the frame.
            update: Whether to overwrite the frame at the specified index, if it
              exists. Default is False.
        """
        if (
            index not in self.frames_dict
            or index in self.frames_dict
            and update
        ):
            self.frames_dict[index] = frame
            self.sensors.update(frame.get_sensors())

    def get_frame(self, index: int) -> Frame:
        """Fetches the Frame object at the specified index.

        Parameters:
            index: Serial index for which to retrieve the Frame.

        Return:
            :class:`Frame`: Frame object at the specified index."""
        if index not in self.frames_dict:
            raise ValueError(
                f"This scene does not have a frame at index {index}"
            )
        return self.frames_dict[index]

    def get_frames(self) -> List[Frame]:
        """Fetches a sorted list of Frames of the scene.

        Returns:
            List[:class:`Frame`]: List of Frames, sorted by index ascending.
        """
        return [
            frame
            for _, frame in sorted(
                self.frames_dict.items(), key=lambda x: x[0]
            )
        ]

    def get_sensors(self) -> List[str]:
        """Fetches all sensor names of the scene.

        Returns:
            List of all sensor names associated with frames in the scene."""
        return list(self.sensors)

    def get_item(self, index: int, sensor_name: str) -> DatasetItem:
        """Fetches the DatasetItem object of the given frame and sensor.

        Parameters:
            index: Serial index of the frame from which to fetch the item.
            sensor_name: Name of the sensor, e.g. "lidar" or "front_cam."

        Returns:
            :class:`DatasetItem`: DatasetItem object of the frame and sensor.
        """
        frame = self.get_frame(index)
        return frame.get_item(sensor_name)

    def get_items_from_sensor(self, sensor_name: str) -> List[DatasetItem]:
        """Fetches all DatasetItem objects of the given sensor.

        Parameters:
            sensor_name: Name of the sensor, e.g. "lidar" or "front_cam."

        Returns:
            List[:class:`DatasetItem`]: List of DatasetItem objects associated
            with the specified sensor.
        """
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

    def get_items(self) -> List[DatasetItem]:
        """Fetches all items in the scene.

        Returns:
            List[:class:`DatasetItem`]: Unordered list of all DatasetItem
            objects in the scene.
        """
        return flatten([frame.get_items() for frame in self.get_frames()])

    def info(self):
        """Fetches information about the scene.

        Returns:
            Payload containing::

                {
                    "reference_id": str,
                    "length": int,
                    "num_sensors": int
                }
        """
        return {
            REFERENCE_ID_KEY: self.reference_id,
            LENGTH_KEY: self.length,
            NUM_SENSORS_KEY: self.num_sensors,
        }

    def validate_frames_dict(self):
        # TODO: make private
        is_continuous = set(list(range(len(self.frames_dict)))) == set(
            self.frames_dict.keys()
        )
        assert (
            is_continuous
        ), "frames must be 0-indexed and continuous (no missing frames)"

    @classmethod
    def from_json(cls, payload: dict):
        """Instantiates scene object from schematized JSON dict payload."""
        frames_payload = payload.get(FRAMES_KEY, [])
        frames = [Frame.from_json(frame) for frame in frames_payload]
        return cls(
            reference_id=payload[REFERENCE_ID_KEY],
            frames=frames,
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        """Serializes scene object to schematized JSON dict."""
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
        """Serializes scene object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass
class LidarScene(Scene):
    """Sequence of lidar pointcloud and camera images over time.

    Nucleus 3D datasets are comprised of LidarScenes, which are sequences of
    lidar pointclouds and camera images over time. These sequences are in turn
    comprised of :class:`Frames <Frame>`.

    By organizing data across multiple sensors over time, LidarScenes make it
    easier to interpret pointclouds, allowing you to see objects move over time
    by clicking through frames and providing context in the form of corresponding
    images.

    You can think of scenes and frames as nested groupings of sensor data across
    time:

    * LidarScene for a given location
        * Frame at timestep 0
            * DatasetItem of pointcloud
            * DatasetItem of front camera image
            * DatasetItem of rear camera image
        * Frame at timestep 1
            * ...
        * ...
    * LidarScene for another location
        * ...

    LidarScenes are uploaded to a :class:`Dataset` with any accompanying
    metadata. Frames do not accept metadata, but each of its constituent
    :class:`DatasetItems <DatasetItem>` does.

    Note: Uploads with a different number of frames/items will error out (only
    on scenes that now differ). Existing scenes are expected to retain the
    same structure, i.e. the same number of frames, and same items per frame.
    If a scene definition is changed (for example, additional frames added) the
    update operation will be ignored. If you would like to alter the structure
    of a scene, please delete the scene and re-upload.

    Parameters:
        reference_id (str): User-specified identifier to reference the scene.
        frames (Optional[List[:class:`Frame`]]): List of frames to be a part of
          the scene. A scene can be created before frames or items have been
          added to it, but must be non-empty when uploading to a :class:`Dataset`.
        metadata (Optional[Dict]): Optional metadata to include with the scene.

    Refer to our `guide to uploading 3D data
    <https://docs.nucleus.scale.com/docs/uploading-3d-data>`_ for more info!
    """

    def __repr__(self) -> str:
        return f"LidarScene(reference_id='{self.reference_id}', frames={self.get_frames()}, metadata={self.metadata})"

    def validate(self):
        # TODO: make private
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


@dataclass
class VideoScene(ABC):
    """Video or sequence of images over time.

    Nucleus video datasets are comprised of VideoScenes. These can be
    comprised of a single video, or a sequence of :class:`DatasetItems <DatasetItem>`
    which are equivalent to frames.

    VideoScenes are uploaded to a :class:`Dataset` with any accompanying
    metadata. Each of :class:`DatasetItems <DatasetItem>` representing a frame
    also accepts metadata.

    Note: Updates with different items will error out (only on scenes that
    now differ). Existing video are expected to retain the same frames, and only
    metadata can be updated. If a video definition is changed (for example,
    additional frames added) the update operation will be ignored. If you would
    like to alter the structure of a video scene, please delete the scene and
    re-upload.

    Parameters:
        reference_id (str): User-specified identifier to reference the scene.
        frame_rate (Optional[int]): Required if uploading items. Frame rate of the video.
        video_location (Optional[str]): Required if not uploading items. The remote URL
            containing the video MP4. Remote formats supported include any URL (``http://``
            or ``https://``) or URIs for AWS S3, Azure, or GCS (i.e. ``s3://``, ``gcs://``).
        items (Optional[List[:class:`DatasetItem`]]): Required if not uploading video_location.
            List of items representing frames, to be a part of the scene. A scene can be created
            before items have been added to it, but must be non-empty when uploading to
            a :class:`Dataset`. A video scene can contain a maximum of 3000 items.
        metadata (Optional[Dict]): Optional metadata to include with the scene.
        upload_to_scale (Optional[bool]): Set this to false in order to use
            `privacy mode <https://nucleus.scale.com/docs/privacy-mode>`_. If using privacy mode
            you must upload both a video_location and items to the VideoScene.

            Setting this to false means the actual data within the video scene will not be
            uploaded to scale meaning that you can send in links that are only accessible
            to certain users, and not to Scale.

    Refer to our `guide to uploading video data
    <https://nucleus.scale.com/docs/uploading-video-data>`_ for more info!
    """

    reference_id: str
    frame_rate: Optional[int] = None
    video_location: Optional[str] = None
    items: List[DatasetItem] = field(default_factory=list)
    metadata: Optional[dict] = field(default_factory=dict)
    upload_to_scale: Optional[bool] = True
    attachment_type: Optional[str] = None

    def __post_init__(self):
        if self.attachment_type:
            warnings.warn(
                "The attachment_type parameter is no longer required and will be deprecated soon.",
                DeprecationWarning,
            )
        if self.metadata is None:
            self.metadata = {}

    def __eq__(self, other):
        return all(
            [
                self.reference_id == other.reference_id,
                self.items == other.items,
                self.video_location == other.video_location,
                self.metadata == other.metadata,
            ]
        )

    @property
    def length(self) -> int:
        """Gets number of items in the scene for videos uploaded with an array of images."""
        assert (
            not self.upload_to_scale or not self.video_location
        ), "Only videos with items have a length"
        return len(self.items)

    def validate(self):
        # TODO: make private
        assert (
            self.items or self.video_location
        ), "Please upload either a video_location or an array of dataset items representing frames"
        if self.upload_to_scale is False:
            assert (
                self.frame_rate > 0
            ), "When using privacy mode frame rate must be at least 1"
            assert (
                self.items and self.length > 0
            ), "When using privacy mode scene must have a list of items of length at least 1"
            for item in self.items:
                assert isinstance(
                    item, DatasetItem
                ), "Each item in a scene must be a DatasetItem object"
                assert (
                    item.image_location is not None
                ), "Each item in a video scene must have an image_location"
                assert (
                    item.upload_to_scale is not False
                ), "Please specify whether to upload to scale in the VideoScene for videos"
        elif self.items:
            assert (
                self.frame_rate > 0
            ), "When uploading an array of items frame rate must be at least 1"
            assert (
                self.length > 0
            ), "When uploading an array of items scene must have a list of items of length at least 1"
            assert (
                not self.video_location
            ), "No video location is accepted when uploading an array of items unless you are using privacy mode"
            for item in self.items:
                assert isinstance(
                    item, DatasetItem
                ), "Each item in a scene must be a DatasetItem object"
                assert (
                    item.image_location is not None
                ), "Each item in a video scene must have an image_location"
                assert (
                    item.upload_to_scale is not False
                ), "Please specify whether to upload to scale in the VideoScene for videos"
        else:
            assert (
                not self.frame_rate
            ), "No frame rate is accepted when uploading a video_location"
            assert (
                not self.items
            ), "No list of items is accepted when uploading a video_location unless you are using privacy mode"

    def add_item(
        self, item: DatasetItem, index: int = None, update: bool = False
    ) -> None:
        """Adds DatasetItem to the specified index for videos uploaded as an array of images.

        Parameters:
            item (:class:`DatasetItem`): Video item to add.
            index: Serial index at which to add the item.
            update: Whether to overwrite the item at the specified index, if it
              exists. Default is False.
        """
        assert (
            not self.upload_to_scale or not self.video_location
        ), "Cannot add item to a video without items"
        if index is None:
            index = len(self.items)
        assert (
            0 <= index <= len(self.items)
        ), f"Video scenes must be contiguous so index must be at least 0 and at most {len(self.items)}."
        if index < len(self.items) and update:
            self.items[index] = item
        else:
            self.items.append(item)

    def get_item(self, index: int) -> DatasetItem:
        """Fetches the DatasetItem at the specified index for videos uploaded as an array of images.

        Parameters:
            index: Serial index for which to retrieve the DatasetItem.

        Return:
            :class:`DatasetItem`: DatasetItem at the specified index."""
        assert (
            not self.upload_to_scale or not self.video_location
        ), "Cannot add item to a video without items"
        if index < 0 or index > len(self.items):
            raise ValueError(
                f"This scene does not have an item at index {index}"
            )
        return self.items[index]

    def get_items(self) -> List[DatasetItem]:
        """Fetches a sorted list of DatasetItems of the scene for videos uploaded as an array of images.

        Returns:
            List[:class:`DatasetItem`]: List of DatasetItems, sorted by index ascending.
        """
        assert (
            not self.upload_to_scale or not self.video_location
        ), "Cannot add item to a video without items"
        return self.items

    def info(self):
        """Fetches information about the video scene.

        Returns:
            Payload containing::

                {
                    "reference_id": str,
                    "length": Optional[int],
                    "frame_rate": int,
                    "video_url": Optional[str],
                }
        """
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
        }
        if self.frame_rate:
            payload[FRAME_RATE_KEY] = self.frame_rate
        if self.video_location:
            payload[VIDEO_URL_KEY] = self.video_location
        if self.items:
            payload[LENGTH_KEY] = self.length
        if self.upload_to_scale:
            payload[UPLOAD_TO_SCALE_KEY] = self.upload_to_scale

        return payload

    @classmethod
    def from_json(cls, payload: dict):
        """Instantiates scene object from schematized JSON dict payload."""
        items_payload = payload.get(FRAMES_KEY, [])
        items = [DatasetItem.from_json(item) for item in items_payload]
        return cls(
            reference_id=payload[REFERENCE_ID_KEY],
            frame_rate=payload.get(FRAME_RATE_KEY, None),
            items=items,
            metadata=payload.get(METADATA_KEY, {}),
            video_location=payload.get(VIDEO_URL_KEY, None),
            upload_to_scale=payload.get(UPLOAD_TO_SCALE_KEY, True),
        )

    def to_payload(self) -> dict:
        """Serializes scene object to schematized JSON dict."""
        self.validate()
        payload: Dict[str, Any] = {
            REFERENCE_ID_KEY: self.reference_id,
        }
        if self.frame_rate:
            payload[FRAME_RATE_KEY] = self.frame_rate
        if self.metadata:
            payload[METADATA_KEY] = self.metadata
        if self.video_location:
            payload[VIDEO_URL_KEY] = self.video_location
        if self.items:
            items_payload = [
                item.to_payload(is_scene=True) for item in self.items
            ]
            payload[FRAMES_KEY] = items_payload
        if self.upload_to_scale is not None:
            payload[UPLOAD_TO_SCALE_KEY] = self.upload_to_scale
        return payload

    def to_json(self) -> str:
        """Serializes scene object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)


def check_all_scene_paths_remote(
    scenes: Union[List[LidarScene], List[VideoScene]]
):
    for scene in scenes:
        if isinstance(scene, VideoScene) and scene.video_location:
            video_location = getattr(scene, VIDEO_LOCATION_KEY)
            if video_location and is_local_path(video_location):
                raise ValueError(
                    f"All paths for videos must be remote, but {scene.video_location} is either "
                    "local, or a remote URL type that is not supported."
                )
        if isinstance(scene, LidarScene) or scene.items:
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
