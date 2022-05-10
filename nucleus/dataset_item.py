import json
import os.path
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence

from .annotation import is_local_path
from .camera_params import CameraParams
from .constants import (
    BACKEND_REFERENCE_ID_KEY,
    CAMERA_PARAMS_KEY,
    IMAGE_URL_KEY,
    METADATA_KEY,
    ORIGINAL_IMAGE_URL_KEY,
    POINTCLOUD_URL_KEY,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    UPLOAD_TO_SCALE_KEY,
    URL_KEY,
)


class DatasetItemType(Enum):
    IMAGE = "image"
    POINTCLOUD = "pointcloud"


@dataclass  # pylint: disable=R0902
class DatasetItem:  # pylint: disable=R0902
    """A dataset item is an image or pointcloud that has associated metadata.

    Note: for 3D data, please include a :class:`CameraParams` object under a key named
    "camera_params" within the metadata dictionary. This will allow for projecting
    3D annotations to any image within a scene.

    Args:
        image_location (Optional[str]): Required if pointcloud_location is not present:
          The location containing the image for the given row of data. This can be a local
          path, or a remote URL. Remote formats supported include any URL (``http://`` or
          ``https://``) or URIs for AWS S3, Azure, or GCS (i.e. ``s3://``, ``gcs://``).

        pointcloud_location (Optional[str]): Required if image_location is not present:
          The remote URL containing the pointcloud JSON. Remote formats supported include
          any URL (``http://`` or ``https://``) or URIs for AWS S3, Azure, or GCS (i.e.
          ``s3://``, ``gcs://``).

        reference_id (Optional[str]): A user-specified identifier to reference the
          item.

        metadata (Optional[dict]): Extra information about the particular
          dataset item. ints, floats, string values will be made searchable in
          the query bar by the key in this dict. For example, ``{"animal":
          "dog"}`` will become searchable via ``metadata.animal = "dog"``.

          Categorical data can be passed as a string and will be treated
          categorically by Nucleus if there are less than 250 unique values in the
          dataset. This means histograms of values in the "Insights" section and
          autocomplete within the query bar.

          Numerical metadata will generate histograms in the "Insights" section,
          allow for sorting the results of any query, and can be used with the
          modulo operator For example: metadata.frame_number % 5 = 0

          All other types of metadata will be visible from the dataset item detail
          view.

          It is important that string and numerical metadata fields are consistent
          - if a metadata field has a string value, then all metadata fields with
          the same key should also have string values, and vice versa for numerical
          metadata.  If conflicting types are found, Nucleus will return an error
          during upload!

          The recommended way of adding or updating existing metadata is to re-run
          the ingestion (dataset.append) with update=True, which will replace any
          existing metadata with whatever your new ingestion run uses. This will
          delete any metadata keys that are not present in the new ingestion run.
          We have a cache based on image_location that will skip the need for a
          re-upload of the images, so your second ingestion will be faster than
          your first.

          For 3D (sensor fusion) data, it is highly recommended to include
          camera intrinsics the metadata of your camera image items. Nucleus
          requires these intrinsics to create visualizations such as cuboid
          projections. Refer to our `guide to uploading 3D data
          <https://nucleus.scale.com/docs/uploading-3d-data>`_ for more
          info.

          .. todo ::
              Shorten this once we have a guide migrated for metadata, or maybe link
              from other places to here.

        upload_to_scale (Optional[bool]): Set this to false in order to use
          `privacy mode <https://nucleus.scale.com/docs/privacy-mode>`_.

          Setting this to false means the actual data within the item will not be
          uploaded to scale meaning that you can send in links that are only accessible
          to certain users, and not to Scale. Skipping upload to Scale is currently only
          implemented for images.
    """

    image_location: Optional[str] = None
    reference_id: str = (
        "DUMMY_VALUE"  # preserve argument ordering for backwards compatibility
    )
    metadata: Optional[dict] = None
    pointcloud_location: Optional[str] = None
    upload_to_scale: Optional[bool] = True

    def __post_init__(self):
        assert self.reference_id != "DUMMY_VALUE", "reference_id is required."
        assert bool(self.image_location) != bool(
            self.pointcloud_location
        ), "Must specify exactly one of the image_location or pointcloud_location parameters"
        if (self.pointcloud_location) and not self.upload_to_scale:
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
        """Instantiates dataset item object from schematized JSON dict payload."""
        image_url = payload.get(IMAGE_URL_KEY, None) or payload.get(
            ORIGINAL_IMAGE_URL_KEY, None
        )
        if BACKEND_REFERENCE_ID_KEY in payload:
            payload[REFERENCE_ID_KEY] = payload[BACKEND_REFERENCE_ID_KEY]
        return cls(
            image_location=image_url,
            pointcloud_location=payload.get(POINTCLOUD_URL_KEY, None),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            upload_to_scale=payload.get(UPLOAD_TO_SCALE_KEY, True),
        )

    def local_file_exists(self):
        # TODO: make private
        return os.path.isfile(self.image_location)

    def to_payload(self, is_scene=False) -> dict:
        """Serializes dataset item object to schematized JSON dict."""
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
            ), "Must specify image_location for DatasetItems not in a LidarScene or VideoScene"
            payload[IMAGE_URL_KEY] = self.image_location
            payload[UPLOAD_TO_SCALE_KEY] = self.upload_to_scale

        return payload

    def to_json(self) -> str:
        """Serializes dataset item object to schematized JSON string."""
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
        if dataset_item.reference_id is None:
            raise ValueError(
                f"Reference ID cannot be None. Encountered DatasetItem with no reference ID:\n{dataset_item}"
            )
        ref_ids.append(dataset_item.reference_id)
    if len(ref_ids) != len(set(ref_ids)):
        duplicates = {
            f"{key}": f"Count: {value}"
            for key, value in Counter(ref_ids).items()
            if value > 1
        }
        raise ValueError(
            f"Duplicate reference IDs found among dataset_items: {duplicates}"
        )
