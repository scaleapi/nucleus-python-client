from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .constants import (
    DATASET_ITEM_ID_KEY,
    GROUND_TRUTH_ANNOTATION_ID_KEY,
    GROUND_TRUTH_ANNOTATION_LABEL_KEY,
    IOU_KEY,
    MODEL_PREDICTION_ID_KEY,
    MODEL_PREDICTION_LABEL_KEY,
    MODEL_RUN_ID_KEY,
    TRUE_POSITIVE_KEY,
)


class ConfusionCategory(Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


def infer_confusion_category(
    true_positive: bool,
    ground_truth_annotation_label: str,
    model_prediction_label: str,
):
    confusion_category = ConfusionCategory.FALSE_NEGATIVE

    if (
        true_positive
        or model_prediction_label == ground_truth_annotation_label
    ):
        confusion_category = ConfusionCategory.TRUE_POSITIVE
    elif model_prediction_label is not None:
        confusion_category = ConfusionCategory.FALSE_POSITIVE

    return confusion_category


@dataclass
class EvaluationMatch:
    """
    EvaluationMatch is a result from a model run evaluation. It can represent a true positive, false positive,
    or false negative.

    The matching only matches the strongest prediction for each annotation, so if there are multiple predictions
    that overlap a single annotation only the one with the highest overlap metric will be matched.

    The model prediction label and the ground truth annotation label can differ for true positives if there is configured
    an allowed_label_mapping for the model run.

    NOTE: There is no iou thresholding applied to these matches, so it is possible to have a true positive with a low
    iou score. If manually rejecting matches remember that a rejected match produces both a false positive and a false
    negative otherwise you'll skew your aggregates.

    Attributes:
        model_run_id (str): The ID of the model run that produced this match.
        model_prediction_id (str): The ID of the model prediction that was matched. None if the match was a false negative.
        ground_truth_annotation_id (str): The ID of the ground truth annotation that was matched. None if the match was a false positive.
        iou (int): The intersection over union score of the match.
        dataset_item_id (str): The ID of the dataset item that was matched.
        confusion_category (ConfusionCategory): The confusion category of the match.
        model_prediction_label (str): The label of the model prediction that was matched. None if the match was a false negative.
        ground_truth_annotation_label (str): The label of the ground truth annotation that was matched. None if the match was a false positive.
    """

    model_run_id: str
    model_prediction_id: Optional[str]  # field is nullable
    ground_truth_annotation_id: Optional[str]  # field is nullable
    iou: float
    dataset_item_id: str
    confusion_category: ConfusionCategory
    model_prediction_label: Optional[str]  # field is nullable
    ground_truth_annotation_label: Optional[str]  # field is nullable

    @classmethod
    def from_json(cls, payload: dict):
        is_true_positive = payload.get(TRUE_POSITIVE_KEY, False)
        model_prediction_label = payload.get(MODEL_PREDICTION_LABEL_KEY, None)
        ground_truth_annotation_label = payload.get(
            GROUND_TRUTH_ANNOTATION_LABEL_KEY, None
        )

        confusion_category = infer_confusion_category(
            true_positive=is_true_positive,
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
