import sys
from abc import abstractmethod
from typing import List, Optional, Union

from nucleus.annotation import AnnotationList, CuboidAnnotation
from nucleus.prediction import CuboidPrediction, PredictionList

from .base import Metric, ScalarResult
from .cuboid_utils import detection_iou, label_match_wrapper, recall_precision
from .filtering import ListOfAndFilters, ListOfOrAndFilters
from .filters import confidence_filter


class CuboidMetric(Metric):
    """Abstract class for metrics of cuboids.

    The CuboidMetric class automatically filters incoming annotations and
    predictions for only cuboid annotations. It also filters
    predictions whose confidence is less than the provided confidence_threshold.
    Finally, it provides support for enforcing matching labels. If
    `enforce_label_match` is set to True, then annotations and predictions will
    only be matched if they have the same label.

    To create a new concrete CuboidMetric, override the `eval` function
    with logic to define a metric between cuboid annotations and predictions.
    """

    def __init__(
        self,
        enforce_label_match: bool = False,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes CuboidMetric abstract object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Default False
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF),
                 like [[MetadataFilter('x', '==', 0), FieldFilter('label', '==', 'pedestrian')], ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single field predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective and multiple column predicate. Finally, the most outer list combines
                these filters as a disjunction (OR).
            prediction_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF),
                 like [[MetadataFilter('x', '==', 0), FieldFilter('label', '==', 'pedestrian')], ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single field predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective and multiple column predicate. Finally, the most outer list combines
                these filters as a disjunction (OR).
        """
        self.enforce_label_match = enforce_label_match
        assert 0 <= confidence_threshold <= 1
        self.confidence_threshold = confidence_threshold
        super().__init__(annotation_filters, prediction_filters)

    @abstractmethod
    def eval(
        self,
        annotations: List[CuboidAnnotation],
        predictions: List[CuboidPrediction],
    ) -> ScalarResult:
        # Main evaluation function that subclasses must override.
        pass

    def aggregate_score(self, results: List[ScalarResult]) -> ScalarResult:  # type: ignore[override]
        return ScalarResult.aggregate(results)

    def call_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> ScalarResult:
        if self.confidence_threshold > 0:
            predictions = confidence_filter(
                predictions, self.confidence_threshold
            )
        cuboid_annotations: List[CuboidAnnotation] = []
        cuboid_annotations.extend(annotations.cuboid_annotations)
        cuboid_predictions: List[CuboidPrediction] = []
        cuboid_predictions.extend(predictions.cuboid_predictions)

        eval_fn = label_match_wrapper(self.eval)
        result = eval_fn(
            cuboid_annotations,
            cuboid_predictions,
            enforce_label_match=self.enforce_label_match,
        )
        return result


class CuboidIOU(CuboidMetric):
    """Calculates the average IOU between cuboid annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        iou_2d: bool = False,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes CuboidIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            iou_2d: whether to return the BEV 2D IOU if true, or the 3D IOU if false.
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter('x', '=', 0), ...], ...]. DNF allows arbitrary boolean logical combinations of single field
                predicates. The innermost structures each describe a single column predicate. The list of inner predicates is
                interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter('x', '=', 0), ...], ...]. DNF allows arbitrary boolean logical combinations of single field
                predicates. The innermost structures each describe a single column predicate. The list of inner predicates is
                interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        self.iou_2d = iou_2d
        super().__init__(
            enforce_label_match=enforce_label_match,
            confidence_threshold=confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    def eval(
        self,
        annotations: List[CuboidAnnotation],
        predictions: List[CuboidPrediction],
    ) -> ScalarResult:
        iou_3d_metric, iou_2d_metric = detection_iou(
            predictions,
            annotations,
            threshold_in_overlap_ratio=self.iou_threshold,
        )

        weight = max(len(annotations), len(predictions))
        if self.iou_2d:
            avg_iou = iou_2d_metric.sum() / max(weight, sys.float_info.epsilon)
        else:
            avg_iou = iou_3d_metric.sum() / max(weight, sys.float_info.epsilon)

        return ScalarResult(avg_iou, weight)


class CuboidPrecision(CuboidMetric):
    """Calculates the average precision between cuboid annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes CuboidIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter('x', '==', 0), ...], ...]. DNF allows arbitrary boolean logical combinations of single field
                predicates. The innermost structures each describe a single column predicate. The list of inner predicates is
                interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter('x', '==', 0), ...], ...]. DNF allows arbitrary boolean logical combinations of single field
                predicates. The innermost structures each describe a single column predicate. The list of inner predicates is
                interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match=enforce_label_match,
            confidence_threshold=confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    def eval(
        self,
        annotations: List[CuboidAnnotation],
        predictions: List[CuboidPrediction],
    ) -> ScalarResult:
        stats = recall_precision(
            predictions,
            annotations,
            threshold_in_overlap_ratio=self.iou_threshold,
        )
        weight = stats["tp_sum"] + stats["fp_sum"]
        precision = stats["tp_sum"] / max(weight, sys.float_info.epsilon)
        return ScalarResult(precision, weight)


class CuboidRecall(CuboidMetric):
    """Calculates the average recall between cuboid annotations and predictions."""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes CuboidIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match=enforce_label_match,
            confidence_threshold=confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    def eval(
        self,
        annotations: List[CuboidAnnotation],
        predictions: List[CuboidPrediction],
    ) -> ScalarResult:
        stats = recall_precision(
            predictions,
            annotations,
            threshold_in_overlap_ratio=self.iou_threshold,
        )
        weight = stats["tp_sum"] + stats["fn_sum"]
        recall = stats["tp_sum"] / max(weight, sys.float_info.epsilon)
        return ScalarResult(recall, weight)
