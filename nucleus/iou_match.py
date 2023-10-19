from dataclasses import dataclass
from enum import Enum

from .constants import (
    DATASET_ITEM_ID_KEY,
    GROUND_TRUTH_ANNOTATION_ID_KEY,
    GROUND_TRUTH_ANNOTATION_LABEL_KEY,
    IOU_KEY,
    MODEL_PREDICTION_ID_KEY,
    MODEL_PREDICTION_LABEL_KEY,
    MODEL_RUN_ID_KEY,
)


class ConfusionCategory(Enum):
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


def infer_confusion_category(
    ground_truth_annotation_label: str,
    model_prediction_label: str,
):
    confusion_category = ConfusionCategory.FALSE_NEGATIVE

    if (
        model_prediction_label is None
        and ground_truth_annotation_label is None
    ):
        confusion_category = ConfusionCategory.TRUE_NEGATIVE
    elif model_prediction_label == ground_truth_annotation_label:
        confusion_category = ConfusionCategory.TRUE_POSITIVE
    elif model_prediction_label is not None:
        confusion_category = ConfusionCategory.FALSE_POSITIVE

    return confusion_category


@dataclass
class IOUMatch:
    model_run_id: str
    model_prediction_id: str | None
    ground_truth_annotation_id: str | None
    iou: int
    dataset_item_id: str
    confusion_category: ConfusionCategory
    model_prediction_label: str | None
    ground_truth_annotation_label: str | None

    @classmethod
    def from_json(cls, payload: dict):
        model_prediction_label = payload.get(MODEL_PREDICTION_LABEL_KEY, None)
        ground_truth_annotation_label = payload.get(
            GROUND_TRUTH_ANNOTATION_LABEL_KEY, None
        )
        confusion_category = infer_confusion_category(
            ground_truth_annotation_label=ground_truth_annotation_label,
            model_prediction_label=model_prediction_label,
        )

        return cls(
            model_run_id=payload[MODEL_RUN_ID_KEY],
            model_prediction_id=payload.get(MODEL_PREDICTION_ID_KEY, None),
            ground_truth_annotation_id=payload.get(
                GROUND_TRUTH_ANNOTATION_ID_KEY, None
            ),
            iou=payload[IOU_KEY],
            dataset_item_id=payload[DATASET_ITEM_ID_KEY],
            confusion_category=confusion_category,
            model_prediction_label=model_prediction_label,
            ground_truth_annotation_label=ground_truth_annotation_label,
        )