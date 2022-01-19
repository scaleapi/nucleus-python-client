from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from sklearn.metrics import f1_score

from nucleus.annotation import AnnotationList, CategoryAnnotation
from nucleus.metrics.base import Metric, MetricResult, ScalarResult
from nucleus.metrics.filters import confidence_filter
from nucleus.prediction import CategoryPrediction, PredictionList

F1_METHODS = {"micro", "macro", "samples", "weighted", "binary"}


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
        # Main evaluation function that subclasses must override.
        # TODO(gunnar): Allow passing multiple predictions and selecting highest confidence? Allows us to show next
        #  contender. Are top-5 scores something that we care about?
        # TODO(gunnar): How do we handle multi-head classification?
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


class CategorizationF1Metric(CategorizationMetric):
    """Evaluation method that matches categories and returns a CategorizationF1Result that aggregates to the F1 score"""

    def __init__(
        self, confidence_threshold: float = 0.0, f1_method: str = "macro"
    ):
        """
        Args:
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            f1_method: {'micro', 'macro', 'samples','weighted', 'binary'}, \
                default='macro'
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        """
        super().__init__(confidence_threshold)
        assert (
            f1_method in F1_METHODS
        ), f"Invalid f1_method {f1_method}, expected one of {F1_METHODS}"
        self.f1_method = f1_method

    def eval(
        self,
        annotations: List[CategoryAnnotation],
        predictions: List[CategoryPrediction],
    ) -> CategorizationResult:
        # TODO(gunnar): Match taxonomy names!
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
        value = f1_score(gt, predicted, average=self.f1_method)
        return ScalarResult(value)
