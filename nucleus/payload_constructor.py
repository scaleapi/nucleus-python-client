from typing import Any, Dict, List, Optional, Union

from .annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)
from .constants import (
    ANNOTATION_METADATA_SCHEMA_KEY,
    ANNOTATION_UPDATE_KEY,
    ANNOTATIONS_KEY,
    ITEMS_KEY,
    LABELS_KEY,
    METADATA_KEY,
    MODEL_BUNDLE_NAME_KEY,
    MODEL_ID_KEY,
    MODEL_TAGS_KEY,
    NAME_KEY,
    REFERENCE_ID_KEY,
    SCENES_KEY,
    SEGMENTATIONS_KEY,
    TAXONOMY_NAME_KEY,
    TYPE_KEY,
    UPDATE_KEY,
)
from .dataset_item import DatasetItem
from .prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
)
from .scene import LidarScene, VideoScene


def construct_append_payload(
    dataset_items: List[DatasetItem], force: bool = False
) -> dict:
    items = []
    for item in dataset_items:
        items.append(item.to_payload())

    return (
        {ITEMS_KEY: items}
        if not force
        else {ITEMS_KEY: items, UPDATE_KEY: True}
    )


def construct_append_scenes_payload(
    scene_list: Union[List[LidarScene], List[VideoScene]],
    update: Optional[bool] = False,
) -> dict:
    scenes = []
    for scene in scene_list:
        scenes.append(scene.to_payload())
    return {SCENES_KEY: scenes, UPDATE_KEY: update}


def construct_annotation_payload(
    annotation_items: List[
        Union[
            BoxAnnotation,
            PolygonAnnotation,
            CuboidAnnotation,
            CategoryAnnotation,
            MultiCategoryAnnotation,
            SegmentationAnnotation,
        ]
    ],
    update: bool,
) -> dict:
    annotations = [
        annotation.to_payload()
        for annotation in annotation_items
        if not isinstance(annotation, SegmentationAnnotation)
    ]
    segmentations = [
        annotation.to_payload()
        for annotation in annotation_items
        if isinstance(annotation, SegmentationAnnotation)
    ]
    payload: Dict[str, Any] = {ANNOTATION_UPDATE_KEY: update}
    if annotations:
        payload[ANNOTATIONS_KEY] = annotations
    if segmentations:
        payload[SEGMENTATIONS_KEY] = segmentations
    return payload


def construct_segmentation_payload(
    annotation_items: Union[
        List[SegmentationAnnotation], List[SegmentationPrediction]
    ],
    update: bool,
) -> dict:
    annotations = []
    for annotation_item in annotation_items:
        annotations.append(annotation_item.to_payload())

    return {SEGMENTATIONS_KEY: annotations, ANNOTATION_UPDATE_KEY: update}


def construct_box_predictions_payload(
    box_predictions: List[
        Union[
            BoxPrediction,
            PolygonPrediction,
            CuboidPrediction,
            CategoryPrediction,
        ]
    ],
    update: bool,
) -> dict:
    predictions = []
    for prediction in box_predictions:
        predictions.append(prediction.to_payload())

    return {ANNOTATIONS_KEY: predictions, ANNOTATION_UPDATE_KEY: update}


def construct_model_creation_payload(
    name: str,
    reference_id: str,
    metadata: Optional[Dict],
    bundle_name: Optional[str],
    tags: Optional[List[str]],
) -> dict:
    payload = {
        NAME_KEY: name,
        REFERENCE_ID_KEY: reference_id,
        METADATA_KEY: metadata if metadata else {},
    }

    if bundle_name:
        payload[MODEL_BUNDLE_NAME_KEY] = bundle_name
    if tags:
        payload[MODEL_TAGS_KEY] = tags

    return payload


def construct_model_run_creation_payload(
    name: str,
    reference_id: Optional[str],
    model_id: Optional[str],
    metadata: Optional[Dict],
    annotation_metadata_schema: Optional[Dict] = None,
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
        ANNOTATION_METADATA_SCHEMA_KEY: annotation_metadata_schema,
    }


def construct_taxonomy_payload(
    taxonomy_name: str, taxonomy_type: str, labels: List[str], update: bool
) -> dict:
    return {
        TAXONOMY_NAME_KEY: taxonomy_name,
        TYPE_KEY: taxonomy_type,
        LABELS_KEY: labels,
        UPDATE_KEY: update,
    }
