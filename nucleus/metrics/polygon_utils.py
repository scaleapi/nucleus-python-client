import sys
from functools import wraps
from typing import Dict, List, Tuple, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from nucleus.annotation import BoxAnnotation, PolygonAnnotation
from nucleus.prediction import BoxPrediction, PolygonPrediction

from .base import ScalarResult
from .errors import PolygonAnnotationTypeError

BoxOrPolygonPrediction = TypeVar(
    "BoxOrPolygonPrediction", BoxPrediction, PolygonPrediction
)
BoxOrPolygonAnnotation = TypeVar(
    "BoxOrPolygonAnnotation", BoxAnnotation, PolygonAnnotation
)


def polygon_annotation_to_shape(
    annotation: BoxOrPolygonAnnotation,
) -> Polygon:
    if isinstance(annotation, BoxAnnotation):
        xmin = annotation.x - annotation.width / 2
        xmax = annotation.x + annotation.width / 2
        ymin = annotation.y - annotation.height / 2
        ymax = annotation.y + annotation.height / 2
        return Polygon(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        )
    elif isinstance(annotation, PolygonAnnotation):
        return Polygon([(point.x, point.y) for point in annotation.vertices])
    else:
        raise PolygonAnnotationTypeError()


def _iou(annotation: Polygon, prediction: Polygon) -> float:
    intersection = annotation.intersection(prediction).area
    union = annotation.area + prediction.area - intersection
    return intersection / max(union, sys.float_info.epsilon)


def _iou_matrix(
    annotations: List[Polygon], predictions: List[Polygon]
) -> np.ndarray:
    iou_matrix = np.empty((len(predictions), len(annotations)))
    for i, prediction in enumerate(predictions):
        for j, annotation in enumerate(annotations):
            iou_matrix[i, j] = _iou(annotation, prediction)
    return iou_matrix


def _iou_assignments_for_same_reference_id(
    annotations: List[BoxOrPolygonAnnotation],
    predictions: List[BoxOrPolygonPrediction],
    iou_threshold: float,
) -> np.ndarray:
    # Check that all annotations and predictions have same reference ID.
    reference_ids = set(annotation.reference_id for annotation in annotations)
    reference_ids |= set(prediction.reference_id for prediction in predictions)
    assert (
        len(reference_ids) <= 1
    ), "Expected annotations and predictions to have same reference ID."

    # Convert annotation and predictions to shapely.geometry.Polygon objects
    polygon_annotations = list(map(polygon_annotation_to_shape, annotations))
    polygon_predictions = list(map(polygon_annotation_to_shape, predictions))

    # Compute IoU matrix and set IoU values below the threshold to 0.
    iou_matrix = _iou_matrix(polygon_annotations, polygon_predictions)
    iou_matrix[iou_matrix < iou_threshold] = 0

    # Match annotations and predictions using linear sum assignment and filter out
    # values below the threshold.
    matched_0, matched_1 = linear_sum_assignment(-iou_matrix)
    iou_assigns = iou_matrix[matched_0, matched_1]
    iou_assigns = iou_assigns[iou_assigns >= iou_threshold]
    return iou_assigns


def group_boxes_or_polygons_by_reference_id(
    annotations: List[BoxOrPolygonAnnotation],
    predictions: List[BoxOrPolygonPrediction],
) -> Dict[
    str, Tuple[List[BoxOrPolygonAnnotation], List[BoxOrPolygonPrediction]]
]:
    """Groups input annotations and predictions by reference_id.

    Args:
        annotations: list of input annotations
        predictions: list of input predictions

    Returns:
        Mapping from each reference_id to (annotations, predictions) tuple.
    """
    reference_ids = set(annotation.reference_id for annotation in annotations)
    reference_ids |= set(prediction.reference_id for prediction in predictions)
    grouped: Dict[
        str, Tuple[List[BoxOrPolygonAnnotation], List[BoxOrPolygonPrediction]]
    ] = {reference_id: ([], []) for reference_id in reference_ids}
    for annotation in annotations:
        grouped[annotation.reference_id][0].append(annotation)
    for prediction in predictions:
        grouped[prediction.reference_id][1].append(prediction)
    return grouped


def group_boxes_or_polygons_by_label(
    annotations: List[BoxOrPolygonAnnotation],
    predictions: List[BoxOrPolygonPrediction],
) -> Dict[
    str, Tuple[List[BoxOrPolygonAnnotation], List[BoxOrPolygonPrediction]]
]:
    """Groups input annotations and predictions by label.

    Args:
        annotations: list of input box or polygon annotations
        predictions: list of input box or polygon predictions

    Returns:
        Mapping from each label to (annotations, predictions) tuple
    """
    labels = set(annotation.label for annotation in annotations)
    labels |= set(prediction.label for prediction in predictions)
    grouped: Dict[
        str, Tuple[List[BoxOrPolygonAnnotation], List[BoxOrPolygonPrediction]]
    ] = {label: ([], []) for label in labels}
    for annotation in annotations:
        grouped[annotation.label][0].append(annotation)
    for prediction in predictions:
        grouped[prediction.label][1].append(prediction)
    return grouped


def iou_assignments(
    annotations: List[BoxOrPolygonAnnotation],
    predictions: List[BoxOrPolygonPrediction],
    iou_threshold: float,
) -> np.ndarray:
    """Matches annotations and predictions based on linear sum cost and returns the
    intersection-over-union values of the matched annotation-prediction pairs, subject
    to the specified IoU threshold. Note that annotations and predictions from
    different reference_ids will not be matched with one another.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    Args:
        annotations: list of box or polygon annotations
        predictions: list of box or polygon predictions
        iou_threshold: the intersection-over-union threshold for an
            annotation-prediction pair to be considered a match.

    Returns:
        1D numpy array that contains the IoU values of the matched pairs.
    """
    grouped_inputs = group_boxes_or_polygons_by_reference_id(
        annotations, predictions
    )
    iou_assigns = []
    for grouped_annotations, grouped_predictions in grouped_inputs.values():
        result_per_reference_id = _iou_assignments_for_same_reference_id(
            grouped_annotations, grouped_predictions, iou_threshold
        )
        iou_assigns.append(result_per_reference_id)
    return np.concatenate(iou_assigns)


def num_true_positives(
    annotations: List[BoxOrPolygonAnnotation],
    predictions: List[BoxOrPolygonPrediction],
    iou_threshold: float,
) -> int:
    """Counts the number of annotations with a matching prediction.

    A prediction is considered a match for an annotation if it has not yet been
    matched to another annotation, its reference_id is the same as the
    annotation, and its IoU with the annotation is at least the iou_threshold.

    Args:
        annotations: list of box or polygon annotations
        predictions: list of box or polygon predictions
        iou_threshold: the intersection-over-union threshold for an
            annotation-prediction pair to be considered a match.

    Returns:
        The number of true positives (predictions that are matched to annotations).
    """
    iou_assigns = iou_assignments(annotations, predictions, iou_threshold)
    true_positives = len(iou_assigns)
    return true_positives


def label_match_wrapper(metric_fn):
    """Decorator to add the ability to only apply metric to annotations and
    predictions with matching labels.

    Args:
        metric_fn: Metric function that takes a list of annotations, a list
            of predictions, and optional args and kwargs.

    Returns:
        Metric function which can optionally enforce matching labels.
    """

    @wraps(metric_fn)
    def wrapper(
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
        *args,
        enforce_label_match: bool = False,
        **kwargs,
    ) -> ScalarResult:
        # Simply return the metric if we are not enforcing label matches.
        if not enforce_label_match:
            return metric_fn(annotations, predictions, *args, **kwargs)

        # For each bin of annotations/predictions, compute the metric applied
        # only to that bin. Then aggregate results across all bins.
        grouped_inputs = group_boxes_or_polygons_by_label(
            annotations, predictions
        )
        metric_results = []
        for binned_annotations, binned_predictions in grouped_inputs.values():
            metric_result = metric_fn(
                binned_annotations, binned_predictions, *args, **kwargs
            )
            metric_results.append(metric_result)
        assert all(
            isinstance(r, ScalarResult) for r in metric_results
        ), "Expected every result to be a ScalarResult"
        return ScalarResult.aggregate(metric_results)

    return wrapper
