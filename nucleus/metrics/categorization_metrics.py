import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

from sklearn.metrics import f1_score

from nucleus.annotation import AnnotationList, CategoryAnnotation
from nucleus.prediction import CategoryPrediction, PredictionList

from .base import Metric, MetricResult, ScalarResult
from .filters import confidence_filter


@dataclass
class CategorizationResult(MetricResult):
    annotation: CategoryAnnotation
    prediction: CategoryPrediction

    @staticmethod
    @abc.abstractmethod
    def aggregate(results: Iterable["CategorizationResult"]) -> "ScalarResult":
        """Aggregates results from all items into a ScalarResult"""


class CategorizationF1Result(CategorizationResult):
    annotation: CategoryAnnotation
    prediction: CategoryPrediction

    @property
    def value(self):
        # TODO: Change task.py interface such that we can return labels
        return 1 if self.annotation.label == self.prediction.label else 0

    @staticmethod
    def aggregate(
        results: Iterable["CategorizationF1Result"],
    ) -> "ScalarResult":
        gt = []
        predicted = []
        for result in results:
            gt.append(result.annotation.label)
            predicted.append(result.prediction.label)
        # TODO(gunnar): Support choice of averaging method
        value = f1_score(gt, predicted, average="macro")
        return ScalarResult(value)


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
    """Evaluation method that matches categories and returns a CategorizationF1Result that aggregates to the F1 score"""

    def eval(
        self,
        annotations: List[CategoryAnnotation],
        predictions: List[CategoryPrediction],
    ) -> CategorizationResult:
        assert (
            len(annotations) == 1
        ), f"Expected only one annotation, got {annotations}"
        assert (
            len(predictions) == 1
        ), f"Expected only one prediction, got {predictions}"
        return CategorizationF1Result(
            annotation=annotations[0], prediction=predictions[0]
        )
