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
    """Quaternion objects are used to represent rotation.
    We use the Hamilton quaternion convention, where i^2 = j^2 = k^2 = ijk = -1, i.e. the right-handed convention.
    The quaternion represented by the tuple (x, y, z, w) is equal to w + x*i + y*j + z*k

    Attributes:
        x: x value
        y: y value
        x: z value
        w: w value
    """

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
    """CameraParams objects represent the camera position/heading used to record the image.

    Attributes:
        position: Vector3 World-normalized position of the camera
        heading: Vector <x, y, z, w> indicating the quaternion of the camera direction;
            note that the z-axis of the camera frame represents the camera's optical axis.
            See `Heading Examples<https://docs.scale.com/reference/data-types-and-the-frame-objects#heading-examples>`_
            for examples.
        fx: focal length in x direction (in pixels)
        fy: focal length in y direction (in pixels)
        cx: principal point x value
        cy: principal point y value
    """

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
    """.. _DatasetItem:
    A dataset item is an image or pointcloud that has associated metadata.

    Note: for 3D data, please include a :class:`.CameraParams` object under a key named
    "camera_params" within the metadata dictionary. This will allow for projecting
    3D annotations to any image within a scene.


    Attributes:
        image_location: Required if pointcloud_location not present: The location
            containing the image for the given row of data. This can be a local path, or a remote URL.
            Remote formats supported include any URL (http:// or https://) or URIs for AWS S3, Azure, or GCS,
            (i.e. s3://, gcs://)
        reference_id: (required) A user-specified identifier to reference the item. The
          default value is present in order to not have to change argument order, but
          must be replaced.
        metadata: Extra information about the particular dataset item. ints, floats,
          string values will be made searchable in the query bar by the key in this dict
          For example, {"animal": "dog"} will become searchable via
          metadata.animal = "dog"

          Categorical data can be passed as a string and will be treated categorically
          by Nucleus if there are less than 250 unique values in the dataset. This means
          histograms of values in the "Insights" section and autocomplete
          within the query bar.

          Numerical metadata will generate histograms in the "Insights" section, allow
          for sorting the results of any query, and can be used with the modulo operator
          For example: metadata.frame_number % 5 = 0

          All other types of metadata will be visible from the dataset item detail view.

          It is important that string and numerical metadata fields are consistent - if
          a metadata field has a string value, then all metadata fields with the same
          key should also have string values, and vice versa for numerical metadata.
          If conflicting types are found, Nucleus will return an error during upload!

          The recommended way of adding or updating existing metadata is to re-run the
          ingestion (dataset.append) with update=True, which will replace any existing
          metadata with whatever your new ingestion run uses. This will delete any
          metadata keys that are not present in the new ingestion run. We have a cache
          based on image_location that will skip the need for a re-upload of the images,
          so your second ingestion will be faster than your first.

          .. todo ::
              Shorten this once we have a guide migrated for metadata, or maybe link
              from other places to here.

        pointcloud_location: Required if image_location not present: The remote URL
          containing the pointcloud JSON. Remote formats supported include any URL
          (http:// or https://) or URIs for AWS S3, Azure, or GCS, (i.e. s3://, gcs://)
        upload_to_scale: Set this to false in order to use `privacy mode <https://dashboard.scale.com/nucleus/docs/api#privacy-mode>`_.

            .. todo ::
                update this once guide is migrated

          Setting this to false means the actual data within the item
          (i.e. the image or pointcloud) will not be uploaded to scale meaning that
          you can send in links that are only accessible to certain users, and not to
          Scale.
    """

    image_location: Optional[str] = None
    reference_id: str = "DUMMY_VALUE"  # Done in order to preserve argument ordering and not break old clients.
    metadata: Optional[dict] = None
    pointcloud_location: Optional[str] = None
    upload_to_scale: Optional[bool] = True

    def __post_init__(self):
        assert self.reference_id != "DUMMY_VALUE", "reference_id is required."
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
    def from_json(cls, payload: dict):
        image_url = payload.get(IMAGE_URL_KEY, None) or payload.get(
            ORIGINAL_IMAGE_URL_KEY, None
        )
        return cls(
            image_location=image_url,
            pointcloud_location=payload.get(POINTCLOUD_URL_KEY, None),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            upload_to_scale=payload.get(UPLOAD_TO_SCALE_KEY, True),
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
