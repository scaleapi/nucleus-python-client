import sys
from abc import abstractmethod
from typing import List, Union

from nucleus.annotation import AnnotationList, BoxAnnotation, PolygonAnnotation
from nucleus.prediction import BoxPrediction, PolygonPrediction, PredictionList

from .base import Metric, ScalarResult
from .filters import confidence_filter
from .polygon_utils import (
    BoxOrPolygonAnnotation,
    BoxOrPolygonPrediction,
    iou_assignments,
    label_match_wrapper,
    num_true_positives,
)


class PolygonMetric(Metric):
    """Abstract class for metrics of box and polygons.

    The PolygonMetric class automatically filters incoming annotations and
    predictions for only box and polygon annotations. It also filters
    predictions whose confidence is less than the provided confidence_threshold.
    Finally, it provides support for enforcing matching labels. If
    `enforce_label_match` is set to True, then annotations and predictions will
    only be matched if they have the same label.

    To create a new concrete PolygonMetric, override the `eval` function
    with logic to define a metric between box/polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation
        from nucleus.metrics import MetricResult, PolygonMetric
        from nucleus.metrics.polygon_utils import BoxOrPolygonAnnotation, BoxOrPolygonPrediction

        class MyPolygonMetric(PolygonMetric):
            def eval(
                self,
                annotations: List[BoxOrPolygonAnnotation],
                predictions: List[BoxOrPolygonPrediction],
            ) -> MetricResult:
                value = (len(annotations) - len(predictions)) ** 2
                weight = len(annotations)
                return MetricResult(value, weight)

        box = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        metric = MyPolygonMetric()
        metric([box], [box])
    """

    def __init__(
        self,
        enforce_label_match: bool = False,
        confidence_threshold: float = 0.0,
    ):
        """Initializes PolygonMetric abstract object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Default False
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        self.enforce_label_match = enforce_label_match
        assert 0 <= confidence_threshold <= 1
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        # Main evaluation function that subclasses must override.
        pass

    def aggregate_score(self, results: List[ScalarResult]) -> ScalarResult:  # type: ignore[override]
        return ScalarResult.aggregate(results)

    def __call__(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> ScalarResult:
        if self.confidence_threshold > 0:
            predictions = confidence_filter(
                predictions, self.confidence_threshold
            )
        polygon_annotations: List[Union[BoxAnnotation, PolygonAnnotation]] = []
        polygon_annotations.extend(annotations.box_annotations)
        polygon_annotations.extend(annotations.polygon_annotations)
        polygon_predictions: List[Union[BoxPrediction, PolygonPrediction]] = []
        polygon_predictions.extend(predictions.box_predictions)
        polygon_predictions.extend(predictions.polygon_predictions)

        eval_fn = label_match_wrapper(self.eval)
        result = eval_fn(
            polygon_annotations,
            polygon_predictions,
            enforce_label_match=self.enforce_label_match,
        )
        return result


class PolygonIOU(PolygonMetric):
    """Calculates the average IOU between box or polygon annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
    ):
        """Initializes PolygonIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(enforce_label_match, confidence_threshold)

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        iou_assigns = iou_assignments(
            annotations, predictions, self.iou_threshold
        )
        weight = max(len(annotations), len(predictions))
        avg_iou = iou_assigns.sum() / max(weight, sys.float_info.epsilon)
        return ScalarResult(avg_iou, weight)


class PolygonPrecision(PolygonMetric):
    """Calculates the precision between box or polygon annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
    ):
        """Initializes PolygonPrecision object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(enforce_label_match, confidence_threshold)

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        true_positives = num_true_positives(
            annotations, predictions, self.iou_threshold
        )
        weight = len(predictions)
        return ScalarResult(
            true_positives / max(weight, sys.float_info.epsilon), weight
        )


class PolygonRecall(PolygonMetric):
    """Calculates the recall between box or polygon annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
    ):
        """Initializes PolygonRecall object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(enforce_label_match, confidence_threshold)

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        true_positives = num_true_positives(
            annotations, predictions, self.iou_threshold
        )
        weight = len(annotations) + sys.float_info.epsilon
        return ScalarResult(
            true_positives / max(weight, sys.float_info.epsilon), weight
        )
