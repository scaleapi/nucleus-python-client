from collections import Counter
import json
import os.path
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from enum import Enum

from .annotation import is_local_path, Point3D
from .constants import (
    IMAGE_URL_KEY,
    METADATA_KEY,
    ORIGINAL_IMAGE_URL_KEY,
    UPLOAD_TO_SCALE_KEY,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    URL_KEY,
    CAMERA_PARAMS_KEY,
    POINTCLOUD_URL_KEY,
    X_KEY,
    Y_KEY,
    Z_KEY,
    W_KEY,
    POSITION_KEY,
    HEADING_KEY,
    FX_KEY,
    FY_KEY,
    CX_KEY,
    CY_KEY,
)


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


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"


@dataclass  # pylint: disable=R0902
class DatasetItem:  # pylint: disable=R0902
    image_location: Optional[str] = None
    reference_id: Optional[str] = None
    metadata: Optional[dict] = None
    pointcloud_location: Optional[str] = None
    upload_to_scale: Optional[bool] = True

    def __post_init__(self):
        assert self.reference_id is not None, "reference_id is required."
        assert bool(self.image_location) != bool(
            self.pointcloud_location
        ), "Must specify exactly one of the image_location, pointcloud_location parameters"
        if self.pointcloud_location and not self.upload_to_scale:
            raise NotImplementedError(
                "Skipping upload to Scale is not currently implemented for pointclouds."
            )
        self.local = (
            is_local_path(self.image_location) if self.image_location else None
        )
        self.type = (
            DatasetItemType.IMAGE
            if self.image_location
            else DatasetItemType.POINTCLOUD
        )
        camera_params = (
            self.metadata.get(CAMERA_PARAMS_KEY, None)
            if self.metadata
            else None
        )
        self.camera_params = (
            CameraParams.from_json(camera_params) if camera_params else None
        )

    @classmethod
    def from_json(cls, payload: dict, is_scene=False):
        image_url = payload.get(IMAGE_URL_KEY, None) or payload.get(
            ORIGINAL_IMAGE_URL_KEY, None
        )

        if is_scene:
            return cls(
                image_location=image_url,
                pointcloud_location=payload.get(POINTCLOUD_URL_KEY, None),
                reference_id=payload.get(REFERENCE_ID_KEY, None),
                metadata=payload.get(METADATA_KEY, {}),
            )

        return cls(
            image_location=image_url,
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            upload_to_scale=payload.get(UPLOAD_TO_SCALE_KEY, None),
        )

    def local_file_exists(self):
        return os.path.isfile(self.image_location)

    def to_payload(self, is_scene=False) -> dict:
        payload: Dict[str, Any] = {
            METADATA_KEY: self.metadata or {},
        }

        payload[REFERENCE_ID_KEY] = self.reference_id

        if is_scene:
            if self.image_location:
                payload[URL_KEY] = self.image_location
            elif self.pointcloud_location:
                payload[URL_KEY] = self.pointcloud_location
            payload[TYPE_KEY] = self.type.value
            if self.camera_params:
                payload[CAMERA_PARAMS_KEY] = self.camera_params.to_payload()
        else:
            assert (
                self.image_location
            ), "Must specify image_location for DatasetItems not in a LidarScene"
            payload[IMAGE_URL_KEY] = self.image_location
            payload[UPLOAD_TO_SCALE_KEY] = self.upload_to_scale

        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


def check_all_paths_remote(dataset_items: Sequence[DatasetItem]):
    for item in dataset_items:
        if item.image_location and is_local_path(item.image_location):
            raise ValueError(
                f"All paths must be remote, but {item.image_location} is either "
                "local, or a remote URL type that is not supported."
            )


def check_for_duplicate_reference_ids(dataset_items: Sequence[DatasetItem]):
    ref_ids = []
    for dataset_item in dataset_items:
        if dataset_item.reference_id is not None:
            ref_ids.append(dataset_item.reference_id)
    if len(ref_ids) != len(set(ref_ids)):
        duplicates = {
            f"{key}": f"Count: {value}"
            for key, value in Counter(ref_ids).items()
        }
        raise ValueError(
            f"Duplicate reference ids found among dataset_items: {duplicates}"
        )
