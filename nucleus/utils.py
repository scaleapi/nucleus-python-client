"""Shared stateless utility function library"""

import io
import json
import uuid
from collections import defaultdict
from typing import IO, Dict, List, Sequence, Type, Union

import requests
from requests.models import HTTPError

from nucleus.annotation import (
    Annotation,
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)

from .constants import (
    ANNOTATION_TYPES,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    CUBOID_TYPE,
    ITEM_KEY,
    MULTICATEGORY_TYPE,
    POLYGON_TYPE,
    REFERENCE_ID_KEY,
    SEGMENTATION_TYPE,
)
from .dataset_item import DatasetItem
from .prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
)
from .scene import LidarScene

STRING_REPLACEMENTS = {
    "\\\\n": "\n",
    "\\\\t": "\t",
    '\\\\"': '"',
}


def format_prediction_response(
    response: dict,
) -> Union[
    dict,
    List[
        Union[
            BoxPrediction,
            PolygonPrediction,
            CuboidPrediction,
            CategoryPrediction,
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
            Type[CuboidPrediction],
            Type[CategoryPrediction],
            Type[SegmentationPrediction],
        ],
    ] = {
        BOX_TYPE: BoxPrediction,
        POLYGON_TYPE: PolygonPrediction,
        CUBOID_TYPE: CuboidPrediction,
        CATEGORY_TYPE: CategoryPrediction,
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
      item_dict: A dictionary with two entries, one for the dataset item, and annother
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


def convert_export_payload(api_payload):
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
            annotations[SEGMENTATION_TYPE] = SegmentationAnnotation.from_json(
                segmentation
            )
        for polygon in row[POLYGON_TYPE]:
            polygon[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[POLYGON_TYPE].append(
                PolygonAnnotation.from_json(polygon)
            )
        for box in row[BOX_TYPE]:
            box[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[BOX_TYPE].append(BoxAnnotation.from_json(box))
        for cuboid in row[CUBOID_TYPE]:
            cuboid[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[CUBOID_TYPE].append(CuboidAnnotation.from_json(cuboid))
        for category in row[CATEGORY_TYPE]:
            category[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[CATEGORY_TYPE].append(
                CategoryAnnotation.from_json(category)
            )
        for multicategory in row[MULTICATEGORY_TYPE]:
            multicategory[REFERENCE_ID_KEY] = row[ITEM_KEY][REFERENCE_ID_KEY]
            annotations[MULTICATEGORY_TYPE].append(
                MultiCategoryAnnotation.from_json(multicategory)
            )
        return_payload_row[ANNOTATIONS_KEY] = annotations
        return_payload.append(return_payload_row)
    return return_payload


def serialize_and_write(
    upload_units: Sequence[Union[DatasetItem, Annotation, LidarScene]],
    file_pointer,
):
    for unit in upload_units:
        try:
            if isinstance(unit, (DatasetItem, Annotation, LidarScene)):
                file_pointer.write(unit.to_json() + "\n")
            else:
                file_pointer.write(json.dumps(unit) + "\n")
        except TypeError as e:
            type_name = type(unit).__name__
            message = (
                f"The following {type_name} could not be serialized: {unit}\n"
            )
            message += (
                "This is usally an issue with a custom python object being "
                "present in the metadata. Please inspect this error and adjust the "
                "metadata so it is json-serializable: only python primitives such as "
                "strings, ints, floats, lists, and dicts. For example, you must "
                "convert numpy arrays into list or lists of lists.\n"
            )
            message += f"The specific error was {e}"
            raise ValueError(message) from e


def upload_to_presigned_url(presigned_url: str, file_pointer: IO):
    # TODO optimize this further to deal with truly huge files and flaky internet connection.
    upload_response = requests.put(presigned_url, file_pointer)
    if not upload_response.ok:
        raise HTTPError(
            f"Tried to put a file to url, but failed with status {upload_response.status_code}. The detailed error was: {upload_response.text}"
        )


def serialize_and_write_to_presigned_url(
    upload_units: Sequence[Union[DatasetItem, Annotation, LidarScene]],
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
