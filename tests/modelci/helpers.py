from typing import List

from nucleus import BoxAnnotation, BoxPrediction, Dataset, DatasetItem, Model


def create_box_annotations(
    dataset: Dataset, dataset_items: List[DatasetItem]
) -> List[BoxAnnotation]:
    annotations = [
        BoxAnnotation(
            label=f"[Pytest] Box Annotation {ds_item.reference_id}",
            x=50 + i * 10,
            y=60 + i * 10,
            width=70 + i * 10,
            height=80 + i * 10,
            reference_id=ds_item.reference_id,
            annotation_id=f"[Pytest] Box Annotation Annotation Id{i}",
        )
        for i, ds_item in enumerate(dataset_items)
    ]
    dataset.append(annotations)
    return annotations


def create_predictions(
    dataset: Dataset, model: Model, annotations: List[BoxAnnotation]
) -> List[BoxPrediction]:
    predictions = [
        BoxPrediction(
            label=ann.label,
            x=ann.x,
            y=ann.y,
            width=ann.width,
            height=ann.height,
            reference_id=ann.reference_id,
            confidence=0.1 * i,
        )
        for i, ann in enumerate(annotations)
    ]
    dataset.upload_predictions(model, predictions)
    return predictions
