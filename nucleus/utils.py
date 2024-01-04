"""Shared stateless utility function library"""
import glob
import io
import json
import os
import uuid
from collections import defaultdict
from typing import IO, TYPE_CHECKING, Dict, List, Sequence, Tuple, Type, Union

import requests
from PIL import Image
from requests.models import HTTPError

from nucleus.annotation import (
    Annotation,
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    KeypointsAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)
from nucleus.errors import NucleusAPIError

from .constants import (
    ANNOTATION_TYPES,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    CUBOID_TYPE,
    EXPORTED_SCALE_TASK_INFO_ROWS,
    ITEM_KEY,
    KEYPOINTS_TYPE,
    LINE_TYPE,
    MAX_PAYLOAD_SIZE,
    MULTICATEGORY_TYPE,
    NEXT_TOKEN_KEY,
    PAGE_SIZE_KEY,
    PAGE_TOKEN_KEY,
    POLYGON_TYPE,
    PREDICTIONS_KEY,
    REFERENCE_ID_KEY,
    SCALE_TASK_INFO_KEY,
    SCENE_KEY,
    SEGMENTATION_TYPE,
)
from .dataset_item import DatasetItem
from .prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    KeypointsPrediction,
    LinePrediction,
    PolygonPrediction,
    SceneCategoryPrediction,
    SegmentationPrediction,
)
from .scene import LidarScene, VideoScene

STRING_REPLACEMENTS = {
    "\\\\n": "\n",
    "\\\\t": "\t",
    '\\\\"': '"',
}

if TYPE_CHECKING:
    from . import NucleusClient


class KeyErrorDict(dict):
    """Wrapper for response dicts with deprecated keys.

    Parameters:
        **kwargs: Mapping from the deprecated key to a warning message.
    """

    def __init__(self, **kwargs):
        self._deprecated = {}

        for key, msg in kwargs.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"All keys must be strings! Received non-string '{key}'"
                )
            if not isinstance(msg, str):
                raise TypeError(
                    f"All warning messages must be strings! Received non-string '{msg}'"
                )

            self._deprecated[key] = msg

        super().__init__()

    def __missing__(self, key):
        """Raises KeyError for deprecated keys, otherwise uses base dict logic."""
        if key in self._deprecated:
            raise KeyError(self._deprecated[key])
        try:
            super().__missing__(key)
        except AttributeError as e:
            raise KeyError(key) from e


def format_prediction_response(
    response: dict,
) -> Union[
    dict,
    List[
        Union[
            BoxPrediction,
            PolygonPrediction,
            LinePrediction,
            KeypointsPrediction,
            CuboidPrediction,
            CategoryPrediction,
            SceneCategoryPrediction,
            SegmentationPrediction,
        ]
    ],
]:
    """Helper function to convert JSON response from endpoints to python objects

    Args:
        response: JSON dictionary response from REST endpoint.
    Returns:
        annotation_response: Dictionary containing a list of annotations for each type,
            keyed by the type name.
    """
    annotation_payload = response.get(ANNOTATIONS_KEY, None)
    if not annotation_payload:
        # An error occurred
        return response
    annotation_response = {}
    type_key_to_class: Dict[
        str,
        Union[
            Type[BoxPrediction],
            Type[PolygonPrediction],
            Type[LinePrediction],
            Type[CuboidPrediction],
            Type[CategoryPrediction],
            Type[KeypointsPrediction],
            Type[SceneCategoryPrediction],
            Type[SegmentationPrediction],
        ],
    ] = {
        BOX_TYPE: BoxPrediction,
        LINE_TYPE: LinePrediction,
        POLYGON_TYPE: PolygonPrediction,
        CUBOID_TYPE: CuboidPrediction,
        CATEGORY_TYPE: CategoryPrediction,
        KEYPOINTS_TYPE: KeypointsPrediction,
        SEGMENTATION_TYPE: SegmentationPrediction,
    }
    for type_key in annotation_payload:
        type_class = type_key_to_class[type_key]
        annotation_response[type_key] = [
            type_class.from_json(annotation)
            for annotation in annotation_payload[type_key]
        ]
    return annotation_response


def format_dataset_item_response(response: dict) -> dict:
    """Format the raw client response into api objects.

    Args:
      response: JSON dictionary response from REST endpoint
    Returns:
      item_dict: A dictionary with two entries, one for the dataset item, and another
        for all of the associated annotations.
    """
    if ANNOTATIONS_KEY not in response:
        raise ValueError(
            f"Server response was missing the annotation key: {response}"
        )
    if ITEM_KEY not in response:
        raise ValueError(
            f"Server response was missing the item key: {response}"
        )
    item = response[ITEM_KEY]
    annotation_payload = response[ANNOTATIONS_KEY]

    annotation_response = {}
    for annotation_type in ANNOTATION_TYPES:
        if annotation_type in annotation_payload:
            annotation_response[annotation_type] = [
                Annotation.from_json(ann)
                for ann in annotation_payload[annotation_type]
            ]
    return {
        ITEM_KEY: DatasetItem.from_json(item),
        ANNOTATIONS_KEY: annotation_response,
    }


def format_scale_task_info_response(response: dict) -> Union[Dict, List[Dict]]:
    """Format the raw client response into api objects.

    Args:
      response: JSON dictionary response from REST endpoint
    Returns:
      A dictionary with two entries, one for the dataset item, and another
        for all of the associated Scale tasks.
    """
    if EXPORTED_SCALE_TASK_INFO_ROWS not in response:
        # Payload is empty so an error occurred
        return response

    ret = []
    for row in response[EXPORTED_SCALE_TASK_INFO_ROWS]:
        if ITEM_KEY in row:
            ret.append(
                {
                    ITEM_KEY: DatasetItem.from_json(row[ITEM_KEY]),
                    SCALE_TASK_INFO_KEY: row[SCALE_TASK_INFO_KEY],
                }
            )
        elif SCENE_KEY in row:
            ret.append(row)
    return ret


# pylint: disable=too-many-branches
def convert_export_payload(api_payload, has_predictions: bool = False):
    """Helper function to convert raw JSON to API objects

    Args:
        api_payload: JSON dictionary response from REST endpoint
    Returns:
        return_payload: A list of dictionaries for each dataset item. Each dictionary
            is in the same format as format_dataset_item_response: one key for the
            dataset item, another for the annotations.
    """
    return_payload = []
    for row in api_payload:
        return_payload_row = {}
        return_payload_row[ITEM_KEY] = DatasetItem.from_json(row[ITEM_KEY])
        annotations = defaultdict(list)
        if row.get(SEGMENTATION_TYPE) is not None:
            segmentation = row[SEGMENTATION_TYPE]
            segmentation[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[
                    SEGMENTATION_TYPE
                ] = SegmentationAnnotation.from_json(segmentation)
            else:
                annotations[
                    SEGMENTATION_TYPE
                ] = SegmentationPrediction.from_json(segmentation)
        for polygon in row[POLYGON_TYPE]:
            polygon[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[POLYGON_TYPE].append(
                    PolygonAnnotation.from_json(polygon)
                )
            else:
                annotations[POLYGON_TYPE].append(
                    PolygonPrediction.from_json(polygon)
                )
        for line in row[LINE_TYPE]:
            line[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[LINE_TYPE].append(LineAnnotation.from_json(line))
            else:
                annotations[LINE_TYPE].append(LinePrediction.from_json(line))
        for keypoints in row[KEYPOINTS_TYPE]:
            keypoints[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[KEYPOINTS_TYPE].append(
                    KeypointsAnnotation.from_json(keypoints)
                )
            else:
                annotations[KEYPOINTS_TYPE].append(
                    KeypointsPrediction.from_json(keypoints)
                )
        for box in row[BOX_TYPE]:
            box[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[BOX_TYPE].append(BoxAnnotation.from_json(box))
            else:
                annotations[BOX_TYPE].append(BoxPrediction.from_json(box))
        for cuboid in row[CUBOID_TYPE]:
            cuboid[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[CUBOID_TYPE].append(
                    CuboidAnnotation.from_json(cuboid)
                )
            else:
                annotations[CUBOID_TYPE].append(
                    CuboidPrediction.from_json(cuboid)
                )
        for category in row[CATEGORY_TYPE]:
            category[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            if not has_predictions:
                annotations[CATEGORY_TYPE].append(
                    CategoryAnnotation.from_json(category)
                )
            else:
                annotations[CATEGORY_TYPE].append(
                    CategoryPrediction.from_json(category)
                )
        for multicategory in row[MULTICATEGORY_TYPE]:
            multicategory[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[MULTICATEGORY_TYPE].append(
                MultiCategoryAnnotation.from_json(multicategory)
            )
        return_payload_row[
            ANNOTATIONS_KEY if not has_predictions else PREDICTIONS_KEY
        ] = annotations
        return_payload.append(return_payload_row)
    return return_payload


def serialize_and_write(
    upload_units: Sequence[
        Union[DatasetItem, Annotation, LidarScene, VideoScene]
    ],
    file_pointer,
):
    """Helper function serialize and write payload to file

    Args:
        upload_units: Sequence of items, annotations or scenes
        file_pointer: Pointer of the file to write to
    """
    bytes_written = 0
    if len(upload_units) == 0:
        raise ValueError(
            "Expecting at least one object when serializing objects to upload, but got zero.  Please try again."
        )
    for unit in upload_units:
        try:
            if isinstance(
                unit, (DatasetItem, Annotation, LidarScene, VideoScene)
            ):
                bytes_written += file_pointer.write(unit.to_json() + "\n")
            else:
                bytes_written += file_pointer.write(json.dumps(unit) + "\n")
        except TypeError as e:
            type_name = type(unit).__name__
            message = (
                f"The following {type_name} could not be serialized: {unit}\n"
            )
            message += (
                "This is usually an issue with a custom python object being "
                "present in the metadata. Please inspect this error and adjust the "
                "metadata so it is json-serializable: only python primitives such as "
                "strings, ints, floats, lists, and dicts. For example, you must "
                "convert numpy arrays into list or lists of lists.\n"
            )
            message += f"The specific error was {e}"
            raise ValueError(message) from e
    if bytes_written > MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"Payload of {bytes_written} bytes exceed maximum payload size of {MAX_PAYLOAD_SIZE} bytes. Please reduce payload size and try again."
        )


def upload_to_presigned_url(presigned_url: str, file_pointer: IO):
    # TODO optimize this further to deal with truly huge files and flaky internet connection.
    upload_response = requests.put(presigned_url, file_pointer)
    if not upload_response.ok:
        raise HTTPError(  # type: ignore
            f"Tried to put a file to url, but failed with status {upload_response.status_code}. The detailed error was: {upload_response.text}"
        )


def serialize_and_write_to_presigned_url(
    upload_units: Sequence[
        Union[DatasetItem, Annotation, LidarScene, VideoScene]
    ],
    dataset_id: str,
    client,
):
    """This helper function can be used to serialize a list of API objects to NDJSON."""
    request_id = uuid.uuid4().hex
    response = client.make_request(
        payload={},
        route=f"dataset/{dataset_id}/signedUrl/{request_id}",
        requests_command=requests.get,
    )

    strio = io.StringIO()
    serialize_and_write(upload_units, strio)
    strio.seek(0)
    upload_to_presigned_url(response["signed_url"], strio)
    return request_id


def replace_double_slashes(s: str) -> str:
    for key, val in STRING_REPLACEMENTS.items():
        s = s.replace(key, val)
    return s


def paginate_generator(
    client: "NucleusClient",
    endpoint: str,
    result_key: str,
    page_size: int = 100000,
    **kwargs,
):
    next_token = None
    while True:
        try:
            response = client.make_request(
                {
                    PAGE_TOKEN_KEY: next_token,
                    PAGE_SIZE_KEY: page_size,
                    **kwargs,
                },
                endpoint,
                requests.post,
            )
        except NucleusAPIError as e:
            if e.status_code == 503:
                e.message += f"/n Your request timed out while trying to get a page size of {page_size}. Try lowering the page_size."
            raise e
        next_token = response[NEXT_TOKEN_KEY]
        for json_value in response[result_key]:
            yield json_value
        if not next_token:
            break


def get_image_dimension(image_fpath: str) -> Tuple[int, int]:
    im = Image.open(image_fpath)
    return im.size


def find_matching_filepaths(
    dirname: str, allowed_file_types: Tuple[str, ...]
) -> List[str]:
    """
    Returns a list of filepaths *relative* to dirname that matched the file globs
    """
    relative_fpaths = []
    for file_type in allowed_file_types:
        pathname = os.path.join(dirname, f"**/*.{file_type}")
        print(f"Searching for filepaths that match {pathname}")
        fpaths = glob.glob(pathname=pathname, recursive=True)
        # keep paths relative to dirname for easier management.
        # TODO: this can be skipped in py version >= 3.10, where `root_dir` can be specified in the glob.
        relative_fpaths.extend(
            [fpath.replace(dirname, "") for fpath in fpaths]
        )
    return relative_fpaths


def create_items_from_folder_crawl(
    dirname: str,
    allowed_file_types: Tuple[str, ...],
    use_privacy_mode: bool,
    privacy_mode_proxy: str,
) -> List[DatasetItem]:
    relative_fpaths = find_matching_filepaths(dirname, allowed_file_types)

    dataset_items = []
    for relative_fpath in relative_fpaths:
        ref_id = relative_fpath

        image_fpath = os.path.join(dirname, relative_fpath)
        width, height = None, None

        if use_privacy_mode:
            width, height = get_image_dimension(image_fpath)
            image_fpath = os.path.join(privacy_mode_proxy, relative_fpath)

        item = DatasetItem(
            image_location=image_fpath,
            reference_id=ref_id,
            width=width,
            height=height,
        )
        dataset_items.append(item)

    return dataset_items
