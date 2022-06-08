import abc
from typing import List, Optional, Union

import fsspec
import numpy as np
from PIL import Image
from s3fs import S3FileSystem

from nucleus.annotation import AnnotationList
from nucleus.metrics.base import MetricResult
from nucleus.metrics.filtering import ListOfAndFilters, ListOfOrAndFilters
from nucleus.metrics.segmentation_utils import (
    instance_mask_to_polys,
    transform_poly_codes_to_poly_preds,
)
from nucleus.prediction import PredictionList

from .base import Metric, ScalarResult
from .polygon_metrics import (
    PolygonAveragePrecision,
    PolygonIOU,
    PolygonMAP,
    PolygonPrecision,
    PolygonRecall,
)


class SegmentationMaskLoader:
    def __init__(self, fs: fsspec):
        self.fs = fs

    def fetch(self, url: str):
        with self.fs.open(url) as fh:
            img = Image.open(fh)
        return img


class SegmentationMaskToPolyMetric(Metric):
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
        self.loader = SegmentationMaskLoader(S3FileSystem(anon=False))

    def call_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> MetricResult:
        assert (
            len(predictions.segmentation_predictions) <= 1
        ), f"Expected only one segmentation mask, got {predictions.segmentation_predictions}"
        prediction = (
            predictions.segmentation_predictions[0]
            if predictions.segmentation_predictions
            else None
        )
        if prediction:
            pred_img = self.loader.fetch(prediction.mask_url)
            pred_value, pred_polys = instance_mask_to_polys(
                np.asarray(pred_img)
            )  # typing: ignore
            code_to_label = {s.index: s.label for s in prediction.annotations}
            poly_predictions = transform_poly_codes_to_poly_preds(
                prediction.reference_id, pred_value, pred_polys, code_to_label
            )
            return self.call_poly_metric(
                annotations,
                PredictionList(polygon_predictions=poly_predictions),
            )
        else:
            return ScalarResult(0, weight=0)

    @abc.abstractmethod
    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        pass


class SegmentationToPolyIOU(SegmentationMaskToPolyMetric):
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
        self.metric = PolygonIOU(
            self.enforce_label_match,
            self.iou_threshold,
            self.confidence_threshold,
            self.annotation_filters,
            self.prediction_filters,
        )

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        return self.metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationToPolyPrecision(SegmentationMaskToPolyMetric):
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
        """Initializes SegmentationToPolyPrecision object.

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
        self.metric = PolygonPrecision(
            self.enforce_label_match,
            self.iou_threshold,
            self.confidence_threshold,
            self.annotation_filters,
            self.prediction_filters,
        )

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        return self.metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationToPolyRecall(SegmentationMaskToPolyMetric):
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
        self.metric = PolygonRecall(
            self.enforce_label_match,
            self.iou_threshold,
            self.confidence_threshold,
            self.annotation_filters,
            self.prediction_filters,
        )

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        return self.metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationToPolyAveragePrecision(SegmentationMaskToPolyMetric):
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
        self.metric = PolygonAveragePrecision(
            self.label,
            self.iou_threshold,
            self.annotation_filters,
            self.prediction_filters,
        )

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        return self.metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationToPolyMAP(SegmentationMaskToPolyMetric):
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
        self.metric = PolygonMAP(
            self.iou_threshold,
            self.annotation_filters,
            self.prediction_filters,
        )

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        return self.metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore
