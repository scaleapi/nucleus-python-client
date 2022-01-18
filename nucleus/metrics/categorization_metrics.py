from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from sklearn.metrics import f1_score

from nucleus.annotation import AnnotationList, CategoryAnnotation
from nucleus.metrics.base import Metric, MetricResult, ScalarResult
from nucleus.metrics.filters import confidence_filter
from nucleus.prediction import CategoryPrediction, PredictionList


@dataclass
class CategorizationResult(MetricResult):
    annotation: CategoryAnnotation
    prediction: CategoryPrediction

    @property
    def value(self):
        # TODO: Change task.py interface such that we can return labels
        return 1 if self.annotation.label == self.prediction.label else 0


class CategorizationMetric(Metric):
    """Abstract class for metrics related to Categorization

    The Categorization class automatically filters incoming annotations and
    predictions for only categorization annotations. It also filters
    predictions whose confidence is less than the provided confidence_threshold.
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
    ) -> CategorizationResult:
        # TODO(gunnar): Allow passing multiple predictions and selecting highest confidence? Allows us to show next
        #  contender. Are top-5 scores something that we care about?
        # Main evaluation function that subclasses must override.
        pass

    @abstractmethod
    def aggregate(self, results: List[CategorizationResult]) -> ScalarResult:  # type: ignore[override]
        pass

    def __call__(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> CategorizationResult:
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
        return CategorizationResult(
            annotation=annotations[0], prediction=predictions[0]
        )

    def aggregate(self, results: List[CategorizationResult]) -> ScalarResult:  # type: ignore[override]
        gt = []
        predicted = []
        for result in results:
            gt.append(result.annotation.label)
            predicted.append(result.prediction.label)
        # TODO(gunnar): Support choice of averaging method
        value = f1_score(gt, predicted, average="macro")
        return ScalarResult(value)
