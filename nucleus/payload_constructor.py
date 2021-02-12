from typing import List, Optional, Dict, Union
from .dataset_item import DatasetItem
from .annotation import BoxAnnotation, PolygonAnnotation
from .prediction import BoxPrediction, PolygonPrediction
from .constants import (
    ANNOTATION_UPDATE_KEY,
    NAME_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
    ANNOTATIONS_KEY,
    ANNOTATION_ITEMS_KEY,
    ITEMS_KEY,
    FORCE_KEY,
    MODEL_ID_KEY,
)


def construct_append_payload(
    dataset_items: List[DatasetItem], force: bool = False
) -> dict:
    items = []
    for item in dataset_items:
        items.append(item.to_payload())

    return (
        {ITEMS_KEY: items}
        if not force
        else {ITEMS_KEY: items, FORCE_KEY: True}
    )


def construct_annotation_payload(
    annotation_items: List[Union[BoxAnnotation, PolygonAnnotation]],
    update: bool,
) -> dict:
    annotations = []
    for annotation_item in annotation_items:
        annotations.append(annotation_item.to_payload())

    return {ANNOTATION_ITEMS_KEY: annotations, ANNOTATION_UPDATE_KEY: update}


def construct_box_predictions_payload(
    box_predictions: List[Union[BoxPrediction, PolygonPrediction]],
) -> dict:
    predictions = []
    for prediction in box_predictions:
        predictions.append(prediction.to_payload())

    return {ANNOTATIONS_KEY: predictions}


def construct_model_creation_payload(
    name: str, reference_id: str, metadata: Optional[Dict]
) -> dict:
    return {
        NAME_KEY: name,
        REFERENCE_ID_KEY: reference_id,
        METADATA_KEY: metadata if metadata else {},
    }


def construct_model_run_creation_payload(
    name: str,
    reference_id: Optional[str],
    model_id: Optional[str],
    metadata: Optional[Dict],
) -> dict:
    payload = {
        NAME_KEY: name,
        METADATA_KEY: metadata if metadata else {},
    }
    if reference_id:
        payload[REFERENCE_ID_KEY] = reference_id
    if model_id:
        payload[MODEL_ID_KEY] = model_id
    return {
        NAME_KEY: name,
        REFERENCE_ID_KEY: reference_id,
        METADATA_KEY: metadata if metadata else {},
    }
