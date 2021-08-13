"""Shared stateless utility function library"""

from collections import defaultdict
import io
import uuid
import json
from typing import IO, Dict, List, Sequence, Union

import requests
from requests.models import HTTPError

from nucleus.annotation import (
    Annotation,
    BoxAnnotation,
    CuboidAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)

from .constants import (
    ANNOTATION_TYPES,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CUBOID_TYPE,
    ITEM_KEY,
    POLYGON_TYPE,
    REFERENCE_ID_KEY,
    SEGMENTATION_TYPE,
)
from .dataset_item import DatasetItem
from .prediction import BoxPrediction, CuboidPrediction, PolygonPrediction
from .scene import LidarScene


def _get_all_field_values(metadata_list: List[dict], key: str):
    return {metadata[key] for metadata in metadata_list if key in metadata}


def suggest_metadata_schema(
    data: Union[
        List[DatasetItem],
        List[BoxPrediction],
        List[PolygonPrediction],
        List[CuboidPrediction],
    ]
):
    metadata_list: List[dict] = [
        d.metadata for d in data if d.metadata is not None
    ]
    schema = {}
    all_keys = {k for metadata in metadata_list for k in metadata.keys()}

    all_key_values: Dict[str, set] = {
        k: _get_all_field_values(metadata_list, k) for k in all_keys
    }

    for key, values in all_key_values.items():
        entry: dict = {}
        if all(isinstance(x, (float, int)) for x in values):
            entry["type"] = "number"
        elif len(values) <= 50:
            entry["type"] = "category"
            entry["choices"] = list(values)
        else:
            entry["type"] = "text"
        schema[key] = entry
    return schema


def format_dataset_item_response(response: dict) -> dict:
    """Format the raw client response into api objects."""
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
