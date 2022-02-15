from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Set, Tuple, Union

from sklearn.metrics import f1_score

from nucleus.annotation import AnnotationList, CategoryAnnotation
from nucleus.metrics.base import Metric, MetricResult, ScalarResult
from nucleus.metrics.filters import confidence_filter
from nucleus.prediction import CategoryPrediction, PredictionList

F1_METHODS = {"micro", "macro", "samples", "weighted", "binary"}


def to_taxonomy_labels(
    anns_or_preds: Union[List[CategoryAnnotation], List[CategoryPrediction]]
) -> Set[str]:
    """Transforms annotation or prediction lists to taxonomy labels by joining them with a seperator (->)"""
    labels = set()
    for item in anns_or_preds:
        taxonomy_label = (
            f"{item.taxonomy_name}->{item.label}"
            if item.taxonomy_name
            else item.label
        )
        labels.add(taxonomy_label)
    return labels


@dataclass
class CategorizationResult(MetricResult):
    annotations: List[CategoryAnnotation]
    predictions: List[CategoryPrediction]

    @property
    def value(self):
        annotation_labels = to_taxonomy_labels(self.annotations)
        prediction_labels = to_taxonomy_labels(self.predictions)

        # TODO: Change task.py interface such that we can return label matching
        # NOTE: Returning 1 if all taxonomy labels match else 0
        value = f1_score(
            annotation_labels, prediction_labels, average=self.f1_method
        )
        return value


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
            confidence_threshold: minimum confidence threshold for predictions to be taken into account for evaluation. Must be in [0, 1]. Default 0.0
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
    def aggregate_score(self, results: List[CategorizationResult]) -> ScalarResult:  # type: ignore[override]
        pass

    def __call__(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> CategorizationResult:
        if self.confidence_threshold > 0:
            predictions = confidence_filter(
                predictions, self.confidence_threshold
            )

        cat_annotations, cat_predictions = self._filter_common_taxonomies(
            annotations.category_annotations, predictions.category_predictions
        )

        result = self.eval(
            cat_annotations,
            cat_predictions,
        )
        return result

    def _filter_common_taxonomies(
        self,
        annotations: List[CategoryAnnotation],
        predictions: List[CategoryPrediction],
    ) -> Tuple[List[CategoryAnnotation], List[CategoryPrediction]]:
        annotated_taxonomies = {ann.taxonomy_name for ann in annotations}
        matching_predictions, matching_taxonomies = self._filter_in_taxonomies(
            predictions, annotated_taxonomies
        )
        matching_annotations, _ = self._filter_in_taxonomies(
            annotations, matching_taxonomies
        )

        return matching_annotations, matching_predictions  # type: ignore

    def _filter_in_taxonomies(
        self,
        anns_or_preds: Union[
            List[CategoryAnnotation], List[CategoryPrediction]
        ],
        filter_on_taxonomies: Set[Union[None, str]],
    ) -> Tuple[
        Union[List[CategoryAnnotation], List[CategoryPrediction]],
        Set[Union[None, str]],
    ]:
        matching_predictions = []
        matching_taxonomies = set()
        for pred in anns_or_preds:
            if pred.taxonomy_name in filter_on_taxonomies:
                matching_predictions.append(pred)
                matching_taxonomies.add(pred.taxonomy_name)
        return matching_predictions, matching_taxonomies


class CategorizationF1(CategorizationMetric):
    """Evaluation method that matches categories and returns a CategorizationF1Result that aggregates to the F1 score"""

    def __init__(
        self, confidence_threshold: float = 0.0, f1_method: str = "macro"
    ):
        """
        Args:
            confidence_threshold: minimum confidence threshold for predictions to be taken into account for evaluation. Must be in [0, 1]. Default 0.0
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
        """
        Notes: This is a little weird eval function. It essentially only does matching of annotation to label and
        the actual metric computation happens in the aggregate step since F1 score only makes sense on a collection.
        """

        return CategorizationResult(
            annotations=annotations, predictions=predictions
        )

    def aggregate_score(self, results: List[CategorizationResult]) -> ScalarResult:  # type: ignore[override]
        gt = []
        predicted = []
        for result in results:
            gt.extend(list(to_taxonomy_labels(result.annotations)))
            predicted.extend(list(to_taxonomy_labels(result.predictions)))
        value = f1_score(gt, predicted, average=self.f1_method)
        return ScalarResult(value)
