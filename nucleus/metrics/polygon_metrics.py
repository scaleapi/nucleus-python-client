import sys
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np

from nucleus.annotation import AnnotationList, BoxAnnotation, PolygonAnnotation
from nucleus.prediction import BoxPrediction, PolygonPrediction, PredictionList

from .base import Metric, ScalarResult
from .custom_types import BoxOrPolygonAnnotation, BoxOrPolygonPrediction
from .filtering import ListOfAndFilters, ListOfOrAndFilters
from .filters import confidence_filter, polygon_label_filter
from .metric_utils import compute_average_precision
from .polygon_utils import (
    get_true_false_positives_confidences,
    group_boxes_or_polygons_by_label,
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

        from typing import List
        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import ScalarResult, PolygonMetric
        from nucleus.metrics.polygon_utils import BoxOrPolygonAnnotation, BoxOrPolygonPrediction

        class MyPolygonMetric(PolygonMetric):
            def eval(
                self,
                annotations: List[BoxOrPolygonAnnotation],
                predictions: List[BoxOrPolygonPrediction],
            ) -> ScalarResult:
                value = (len(annotations) - len(predictions)) ** 2
                weight = len(annotations)
                return ScalarResult(value, weight)

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = MyPolygonMetric()
        metric(annotations, predictions)
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
        """Initializes PolygonMetric abstract object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Default False
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        super().__init__(annotation_filters, prediction_filters)
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

    def call_metric(
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
    """Calculates the average IOU between box or polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import PolygonIOU

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = PolygonIOU()
        metric(annotations, predictions)
    """

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match,
            confidence_threshold,
            annotation_filters,
            prediction_filters,
        )

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
    """Calculates the precision between box or polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import PolygonPrecision

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = PolygonPrecision()
        metric(annotations, predictions)
    """

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonPrecision object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match,
            confidence_threshold,
            annotation_filters,
            prediction_filters,
        )

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
    """Calculates the recall between box or polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import PolygonRecall

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = PolygonRecall()
        metric(annotations, predictions)
    """

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonRecall object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match,
            confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

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


class PolygonAveragePrecision(PolygonMetric):
    """Calculates the average precision between box or polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import PolygonAveragePrecision

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = PolygonAveragePrecision(label="car")
        metric(annotations, predictions)
    """

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        label,
        iou_threshold: float = 0.5,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonRecall object.

        Args:
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        self.label = label
        super().__init__(
            enforce_label_match=False,
            confidence_threshold=0,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        annotations_filtered = polygon_label_filter(annotations, self.label)
        predictions_filtered = polygon_label_filter(predictions, self.label)
        (
            true_false_positives,
            confidences,
        ) = get_true_false_positives_confidences(
            annotations_filtered, predictions_filtered, self.iou_threshold
        )
        if np.all(confidences):
            idxes = np.argsort(-confidences)
            true_false_positives_sorted = true_false_positives[idxes]
        else:
            true_false_positives_sorted = true_false_positives
        cumulative_true_positives = np.cumsum(true_false_positives_sorted)
        total_predictions = np.arange(1, len(true_false_positives) + 1)
        precisions = cumulative_true_positives / total_predictions
        recalls = cumulative_true_positives / len(annotations)
        average_precision = compute_average_precision(precisions, recalls)
        weight = 1
        return ScalarResult(average_precision, weight)


class PolygonMAP(PolygonMetric):
    """Calculates the mean average precision between box or polygon annotations and predictions.
    ::

        from nucleus import BoxAnnotation, Point, PolygonPrediction
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import PolygonMAP

        box_anno = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        polygon_pred = PolygonPrediction(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            confidence=0.8,
            metadata={"vehicle_color": "yellow"}
        )

        annotations = AnnotationList(box_annotations=[box_anno])
        predictions = PredictionList(polygon_predictions=[polygon_pred])
        metric = PolygonMAP()
        metric(annotations, predictions)
    """

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        iou_threshold: float = 0.5,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonRecall object.

        Args:
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        assert (
            0 <= iou_threshold <= 1
        ), "IoU threshold must be between 0 and 1."
        self.iou_threshold = iou_threshold
        super().__init__(
            enforce_label_match=False,
            confidence_threshold=0,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    def eval(
        self,
        annotations: List[BoxOrPolygonAnnotation],
        predictions: List[BoxOrPolygonPrediction],
    ) -> ScalarResult:
        grouped_inputs = group_boxes_or_polygons_by_label(
            annotations, predictions
        )
        results: List[ScalarResult] = []
        for label, group in grouped_inputs.items():
            annotations_group, predictions_group = group
            metric = PolygonAveragePrecision(label)
            result = metric.eval(annotations_group, predictions_group)
            results.append(result)
        average_result = ScalarResult.aggregate(results)
        return average_result
