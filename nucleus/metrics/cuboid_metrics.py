import sys
from abc import abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Callable, List, Optional, Union

from nucleus.annotation import Annotation, AnnotationList, CuboidAnnotation
from nucleus.prediction import CuboidPrediction, Prediction, PredictionList

from .base import Metric, ScalarResult
from .cuboid_utils import detection_iou, label_match_wrapper, recall_precision
from .filters import confidence_filter


class FilterOp(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="


MetadataFilter = namedtuple("MetadataFilter", ["key", "op", "value"])
DNFMetadataFilters = List[List[MetadataFilter]]
DNFMetadataFilters.__doc__ = """\
Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single column predicate. The list of inner predicates is
interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
Finally, the most outer list combines these filters as a disjunction (OR).
"""


def filter_to_comparison_function(
    metadata_filter: MetadataFilter,
) -> Callable[[Union[Annotation, Prediction]], bool]:
    op = FilterOp(metadata_filter.op)
    if op is FilterOp.GT:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            > metadata_filter.value
        )
    elif op is FilterOp.GTE:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            >= metadata_filter.value
        )
    elif op is FilterOp.LT:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            < metadata_filter.value
        )
    elif op is FilterOp.LTE:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            <= metadata_filter.value
        )
    elif op is FilterOp.EQ:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            == metadata_filter.value
        )
    elif op is FilterOp.NEQ:
        return (
            lambda ann_or_pred: ann_or_pred.metadata[metadata_filter.key]
            != metadata_filter.value
        )
    else:
        raise RuntimeError(
            f"Fell through all op cases, no match for: '{op}' - MetadataFilter: {metadata_filter},"
        )


def filter_by_metadata_fields(
    ann_or_pred: Union[List[Annotation], List[Prediction]],
    metadata_filter: Union[DNFMetadataFilters, List[MetadataFilter]],
):
    """
    Attributes:
        ann_or_pred: Prediction or Annotation
        metadata_filter: MetadataFilter predicates. Predicates are expressed in disjunctive normal form (DNF), like
            [[MetadataFilter('x', '=', 0), ...], ...]. DNF allows arbitrary boolean logical combinations of single field
            predicates. The innermost structures each describe a single column predicate. The list of inner predicates is
            interpreted as a conjunction (AND), forming a more selective and multiple column predicate.
            Finally, the most outer list combines these filters as a disjunction (OR).
    """
    if metadata_filter is None or len(metadata_filter) == 0:
        return ann_or_pred

    if isinstance(metadata_filter[0], MetadataFilter):
        # Normalize into DNF
        metadata_filter: DNFMetadataFilters = [metadata_filter]

    filtered = []
    for item in ann_or_pred:
        for or_branch in metadata_filter:
            and_conditions = (
                filter_to_comparison_function(cond) for cond in or_branch
            )
            if all(c(item) for c in and_conditions):
                filtered.append(item)
                break
    return filtered


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
        annotation_filters: Optional[DNFMetadataFilters] = None,
        prediction_filters: Optional[DNFMetadataFilters] = None,
    ):
        """Initializes CuboidMetric abstract object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Default False
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
        self.enforce_label_match = enforce_label_match
        assert 0 <= confidence_threshold <= 1
        self.confidence_threshold = confidence_threshold
        self.annotation_filters = annotation_filters
        self.prediction_filters = prediction_filters

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

    def __call__(
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
        cuboid_annotations = filter_by_metadata_fields(
            cuboid_annotations, self.annotation_filters
        )
        cuboid_predictions = filter_by_metadata_fields(
            cuboid_annotations, self.prediction_filters
        )
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
        annotation_filters: Optional[DNFMetadataFilters] = None,
        prediction_filters: Optional[DNFMetadataFilters] = None,
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
        annotation_filters: Optional[DNFMetadataFilters] = None,
        prediction_filters: Optional[DNFMetadataFilters] = None,
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
        annotation_filters: Optional[DNFMetadataFilters] = None,
        prediction_filters: Optional[DNFMetadataFilters] = None,
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
