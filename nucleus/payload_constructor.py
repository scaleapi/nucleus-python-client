from typing import List
from .dataset_item import DatasetItem
from .annotation import BoxAnnotation
from .prediction import BoxPrediction

def construct_append_payload(dataset_items: List[DatasetItem]) -> dict:
    payload = {}
    items = []
    for item in dataset_items:
        items.append(item.to_payload())

    return {"items": items} 

def construct_box_annotation_payload(box_annotations: List[BoxAnnotation]) -> dict:
    payload = {}
    annotations = []
    for annotation in box_annotations:
        annotations.append(annotation.to_payload())

    return {"annotations": annotations}

def construct_box_predictions_payload(box_predictions: List[BoxPrediction]) -> dict:
    payload = {}
    predictions = []
    for prediction in box_predictions:
        predictions.append(prediction.to_payload())

    return {"annotations": predictions}

def construct_model_creation_payload(name: str, reference_id: str, metadata: dict) -> dict:
    return {
        "name": name,
        "reference_id": reference_id,
        "metadata": metadata
    }