from typing import Any, Dict, List, Optional, Union

from .annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SceneCategoryAnnotation,
    SegmentationAnnotation,
)
from .constants import (
    ANNOTATION_METADATA_SCHEMA_KEY,
    ANNOTATION_UPDATE_KEY,
    ANNOTATIONS_KEY,
    LABELS_KEY,
    METADATA_KEY,
    MODEL_ARCHITECTURE_KEY,
    MODEL_BUMP_TYPE_KEY,
    MODEL_BUNDLE_NAME_KEY,
    MODEL_DESCRIPTION_KEY,
    MODEL_ID_KEY,
    MODEL_INPUT_SCHEMA_KEY,
    MODEL_NUM_PARAMETERS_KEY,
    MODEL_OUTPUT_SCHEMA_KEY,
    MODEL_PARENT_MODEL_PROJECT_ID_KEY,
    MODEL_TAGS_KEY,
    MODEL_TRAINED_SLICE_IDS_KEY,
    MODEL_TRAINING_DATA_FIELDS_METADATA_KEY,
    MODEL_TRAINING_DATA_KEY,
    MODEL_VERSION_LABEL_KEY,
    MODEL_VERSION_MAJOR_KEY,
    MODEL_VERSION_MINOR_KEY,
    NAME_KEY,
    REFERENCE_ID_KEY,
    SEGMENTATIONS_KEY,
    TAXONOMY_NAME_KEY,
    TRAINED_SLICE_ID_KEY,
    TYPE_KEY,
    UPDATE_KEY,
)
from .prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SceneCategoryPrediction,
    SegmentationPrediction,
)


def construct_annotation_payload(
    annotation_items: List[
        Union[
            BoxAnnotation,
            PolygonAnnotation,
            CuboidAnnotation,
            CategoryAnnotation,
            MultiCategoryAnnotation,
            SceneCategoryAnnotation,
            SegmentationAnnotation,
        ]
    ],
    update: bool,
    trained_slice_id: Optional[str],
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
    if trained_slice_id:
        payload[TRAINED_SLICE_ID_KEY] = trained_slice_id
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
            SceneCategoryPrediction,
        ]
    ],
    update: bool,
) -> dict:
    predictions = []
    for prediction in box_predictions:
        predictions.append(prediction.to_payload())

    return {ANNOTATIONS_KEY: predictions, ANNOTATION_UPDATE_KEY: update}


# Sentinel distinguishing "leave unchanged" from an explicit ``None`` ("clear this
# field") in model update calls, where both are meaningful.
NO_UPDATE = object()


def merge_training_data_fields_into_metadata(
    metadata: Optional[Dict], training_data_fields: Optional[Dict[str, str]]
) -> Optional[Dict]:
    """Return a copy of ``metadata`` with structured training-data fields written under
    the reserved key the dashboard and backend agree on.

    ``training_data_fields`` is a ``{key: value}`` mapping; it is stored as a list of
    ``{"key", "value"}`` objects (the shape the model-registry UI reads/writes and the
    server denormalizes into ``nucleus.model_training_data_field`` for search). Returns
    ``metadata`` unchanged when ``training_data_fields`` is ``None``.
    """
    if training_data_fields is None:
        return metadata
    merged = dict(metadata) if metadata else {}
    merged[MODEL_TRAINING_DATA_FIELDS_METADATA_KEY] = [
        {"key": str(k), "value": str(v)}
        for k, v in training_data_fields.items()
    ]
    return merged


def construct_model_creation_payload(
    name: str,
    reference_id: str,
    metadata: Optional[Dict],
    bundle_name: Optional[str],
    tags: Optional[List[str]],
    trained_slice_ids: Optional[List[str]],
    description: Optional[str] = None,
    architecture: Optional[str] = None,
    num_parameters: Optional[str] = None,
    training_data: Optional[str] = None,
    input_schema: Optional[Dict] = None,
    output_schema: Optional[Dict] = None,
    parent_model_project_id: Optional[str] = None,
    bump_type: Optional[str] = None,
    version_major: Optional[int] = None,
    version_minor: Optional[int] = None,
    version_label: Optional[str] = None,
) -> dict:
    payload = {
        NAME_KEY: name,
        REFERENCE_ID_KEY: reference_id,
        METADATA_KEY: metadata if metadata else {},
    }

    if trained_slice_ids:
        payload[MODEL_TRAINED_SLICE_IDS_KEY] = trained_slice_ids
    if bundle_name:
        payload[MODEL_BUNDLE_NAME_KEY] = bundle_name
    if tags:
        payload[MODEL_TAGS_KEY] = tags

    # Optional descriptive + versioning fields: only sent when provided so the server
    # keeps its defaults for anything omitted.
    optional_fields = {
        MODEL_DESCRIPTION_KEY: description,
        MODEL_ARCHITECTURE_KEY: architecture,
        MODEL_NUM_PARAMETERS_KEY: num_parameters,
        MODEL_TRAINING_DATA_KEY: training_data,
        MODEL_INPUT_SCHEMA_KEY: input_schema,
        MODEL_OUTPUT_SCHEMA_KEY: output_schema,
        MODEL_PARENT_MODEL_PROJECT_ID_KEY: parent_model_project_id,
        MODEL_BUMP_TYPE_KEY: bump_type,
        MODEL_VERSION_MAJOR_KEY: version_major,
        MODEL_VERSION_MINOR_KEY: version_minor,
        MODEL_VERSION_LABEL_KEY: version_label,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value

    return payload


def construct_model_update_payload(
    name=NO_UPDATE,
    reference_id=NO_UPDATE,
    metadata=NO_UPDATE,
    description=NO_UPDATE,
    architecture=NO_UPDATE,
    num_parameters=NO_UPDATE,
    training_data=NO_UPDATE,
    input_schema=NO_UPDATE,
    output_schema=NO_UPDATE,
) -> dict:
    """Build the body for ``POST model/{id}/update``.

    Only fields the caller actually passed (value is not :data:`NO_UPDATE`) are included,
    so unspecified fields are left untouched server-side. Passing ``None`` for a nullable
    field is a deliberate "clear it" and is forwarded as ``null``.
    """
    candidates = {
        NAME_KEY: name,
        REFERENCE_ID_KEY: reference_id,
        METADATA_KEY: metadata,
        MODEL_DESCRIPTION_KEY: description,
        MODEL_ARCHITECTURE_KEY: architecture,
        MODEL_NUM_PARAMETERS_KEY: num_parameters,
        MODEL_TRAINING_DATA_KEY: training_data,
        MODEL_INPUT_SCHEMA_KEY: input_schema,
        MODEL_OUTPUT_SCHEMA_KEY: output_schema,
    }
    return {
        key: value
        for key, value in candidates.items()
        if value is not NO_UPDATE
    }


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
