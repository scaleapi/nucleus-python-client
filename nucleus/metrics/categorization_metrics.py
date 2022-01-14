from abc import abstractmethod
from typing import List

from nucleus.annotation import AnnotationList, CategoryAnnotation
from nucleus.prediction import CategoryPrediction, PredictionList

from .base import Metric, ScalarResult
from .filters import confidence_filter
from .polygon_utils import BoxOrPolygonAnnotation, BoxOrPolygonPrediction


class CategorizationMetric(Metric):
    """Abstract class for metrics related to Categorization

    The Categorization class automatically filters incoming annotations and
    predictions for only categorization annotations. It also filters
    predictions whose confidence is less than the provided confidence_threshold.

    To create a new concrete PolygonMetric, override the `eval` function
    with logic to define a metric between box/polygon annotations and predictions.
    ::

        from nucleus import CategoryAnnotation, CategoryPrediction
        from nucleus.metrics import MetricResult, CategorizationMetric

        class MyCategorizationMetric(CategorizationMetric):
            def eval(
                self,
                annotations: List[CategoryAnnotation],
                predictions: List[CategoryPrediction],
            ) -> MetricResult:
                value = (len(annotations) - len(predictions)) ** 2
                weight = len(annotations)
                return MetricResult(value, weight)

        car_annotation = CategoryAnnotation(
            label="car",
            reference_id="image_1",
            metadata={"vehicle_color": "red"}
        )

        prediction = CategoryPrediction(
            label="car",
            reference_id="image_1",
            confidence=0.9
        )

        metric = MyCategorizationMetric()
        metric([car_annotation], [box])
    """

    def __init__(
        self,
        confidence_threshold: float = 0.0,
    ):
        """Initializes CategorizationMetric abstract object.

        Args:
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        assert 0 <= confidence_threshold <= 1
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def eval(
        self,
        annotations: List[
            CategoryAnnotation
        ],  # TODO(gunnar): List to conform with other APIs or single instance?
        predictions: List[CategoryPrediction],
    ) -> ScalarResult:
        # TODO(gunnar): Allow passing multiple predictions and selecting highest confidence? Allows us to show next
        #  contender. Are top-5 scores something that we care about?
        # Main evaluation function that subclasses must override.
        pass

    def __call__(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> ScalarResult:
        if self.confidence_threshold > 0:
            predictions = confidence_filter(
                predictions, self.confidence_threshold
            )

        result = self.eval(
            annotations.category_annotations,
            predictions.category_predictions,
        )
        return result


class CategorizationF1(CategorizationMetric):
    """Calculates the average IOU between box or polygon annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        confidence_threshold: float = 0.0,
    ):
        """Initializes PolygonIOU object.

        Args:
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        super().__init__(confidence_threshold)

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        # TODO
        pass
