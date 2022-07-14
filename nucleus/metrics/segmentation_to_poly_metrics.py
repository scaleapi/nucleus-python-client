import abc
import logging
from enum import Enum
from typing import List, Optional, Union

import numpy as np

from nucleus.annotation import AnnotationList, SegmentationAnnotation
from nucleus.metrics.base import MetricResult
from nucleus.metrics.filtering import (
    ListOfAndFilters,
    ListOfOrAndFilters,
    apply_filters,
)
from nucleus.metrics.segmentation_utils import (
    instance_mask_to_polys,
    rasterize_polygons_to_segmentation_mask,
    setup_iou_thresholds,
    transform_poly_codes_to_poly_preds,
)
from nucleus.prediction import PredictionList

from .segmentation_loader import InMemoryLoader, SegmentationMaskLoader
from .segmentation_metrics import (
    SegmentationIOU,
    SegmentationMAP,
    SegmentationPrecision,
    SegmentationRecall,
)

try:
    from s3fs import S3FileSystem
except (ModuleNotFoundError, OSError):
    from ..package_not_installed import PackageNotInstalled

    S3FileSystem = PackageNotInstalled

from .base import Metric, ScalarResult
from .polygon_metrics import (
    PolygonAveragePrecision,
    PolygonIOU,
    PolygonPrecision,
    PolygonRecall,
)


class SegToPolyMode(str, Enum):
    GENERATE_GT_FROM_POLY = "gt_from_poly"
    GENERATE_PRED_POLYS_FROM_MASK = "gt_from_poly"


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
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
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
        # Since segmentation annotations are very different from everything else we can't rely on the upper filtering
        super().__init__(None, None)
        self._annotation_filters = annotation_filters
        self._prediction_filters = prediction_filters
        self.enforce_label_match = enforce_label_match
        assert 0 <= confidence_threshold <= 1
        self.confidence_threshold = confidence_threshold
        self.loader = SegmentationMaskLoader(S3FileSystem(anon=False))
        self.mode = mode

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
        annotations.polygon_annotations = apply_filters(
            annotations.polygon_annotations, self._annotation_filters  # type: ignore
        )
        annotations.box_annotations = apply_filters(
            annotations.box_annotations, self._annotation_filters  # type: ignore
        )
        predictions.segmentation_predictions = apply_filters(
            predictions.segmentation_predictions, self._prediction_filters  # type: ignore
        )
        if prediction:
            if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
                pred_img = self.loader.fetch(prediction.mask_url)
                ann_img, segments = rasterize_polygons_to_segmentation_mask(
                    annotations.polygon_annotations
                    + annotations.box_annotations,  # type:ignore
                    pred_img.shape,
                )
                # TODO: apply Segmentation filters after?
                annotations.segmentation_annotations = [
                    SegmentationAnnotation(
                        "__no_url",
                        annotations=segments,
                        reference_id=annotations.polygon_annotations[
                            0
                        ].reference_id,
                    )
                ]
                return self.call_segmentation_metric(
                    annotations,
                    np.asarray(ann_img),
                    predictions,
                    np.asarray(pred_img),
                )
            elif self.mode == SegToPolyMode.GENERATE_PRED_POLYS_FROM_MASK:
                pred_img = self.loader.fetch(prediction.mask_url)
                pred_value, pred_polys = instance_mask_to_polys(
                    np.asarray(pred_img)
                )  # typing: ignore
                code_to_label = {
                    s.index: s.label for s in prediction.annotations
                }
                poly_predictions = transform_poly_codes_to_poly_preds(
                    prediction.reference_id,
                    pred_value,
                    pred_polys,
                    code_to_label,
                )
                return self.call_poly_metric(
                    annotations,
                    PredictionList(polygon_predictions=poly_predictions),
                )
            else:
                raise RuntimeError(
                    f"Misonconfigured class. Got mode '{self.mode}', expected one of {list(SegToPolyMode)}"
                )
        else:
            return ScalarResult(0, weight=0)

    def call_segmentation_metric(
        self,
        annotations: AnnotationList,
        ann_img: np.ndarray,
        predictions: PredictionList,
        pred_img: np.ndarray,
    ):
        metric = self.configure_metric()
        metric.loader = InMemoryLoader(
            {
                annotations.segmentation_annotations[0].mask_url: ann_img,
                predictions.segmentation_predictions[0].mask_url: pred_img,
            }
        )
        return metric(annotations, predictions)

    def call_poly_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ):
        metric = self.configure_metric()
        return metric(annotations, predictions)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        metric = self.configure_metric()
        return metric.aggregate_score(results)  # type: ignore

    @abc.abstractmethod
    def configure_metric(self):
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
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
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
            mode,
        )

    def configure_metric(self):
        if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
            metric = SegmentationIOU(
                self.annotation_filters,
                self.prediction_filters,
                self.iou_threshold,
            )
        else:
            metric = PolygonIOU(
                self.enforce_label_match,
                self.iou_threshold,
                self.confidence_threshold,
                self.annotation_filters,
                self.prediction_filters,
            )
        return metric


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
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
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
            mode,
        )

    def configure_metric(self):
        if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
            metric = SegmentationPrecision(
                self.annotation_filters,
                self.prediction_filters,
                self.iou_threshold,
            )
        else:
            metric = PolygonPrecision(
                self.enforce_label_match,
                self.iou_threshold,
                self.confidence_threshold,
                self.annotation_filters,
                self.prediction_filters,
            )
        return metric


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
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
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
            annotation_filters,
            prediction_filters,
            mode,
        )

    def configure_metric(self):
        if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
            metric = SegmentationRecall(
                self.annotation_filters,
                self.prediction_filters,
                self.iou_threshold,
            )
        else:
            metric = PolygonRecall(
                self.enforce_label_match,
                self.iou_threshold,
                self.confidence_threshold,
                self.annotation_filters,
                self.prediction_filters,
            )
        return metric


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
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
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

    def configure_metric(self):
        if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
            # TODO(gunnar): Add a label filter
            metric = SegmentationPrecision(
                self.annotation_filters,
                self.prediction_filters,
                self.iou_threshold,
            )
        else:
            metric = PolygonAveragePrecision(
                self.label,
                self.iou_threshold,
                self.annotation_filters,
                self.prediction_filters,
            )
        return metric


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
        iou_threshold: float = -1,
        iou_thresholds: Union[List[float], str] = "coco",
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        mode: SegToPolyMode = SegToPolyMode.GENERATE_GT_FROM_POLY,
    ):
        """Initializes PolygonRecall object.

        Args:
            iou_thresholds: IOU thresholds to check AP at
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
        if iou_threshold:
            logging.warning(
                "Got deprecated parameter 'iou_threshold'. Ignoring it."
            )
        self.iou_thresholds = setup_iou_thresholds(iou_thresholds)
        super().__init__(
            False, 0, annotation_filters, prediction_filters, mode
        )

    def configure_metric(self):
        if self.mode == SegToPolyMode.GENERATE_GT_FROM_POLY:
            # TODO(gunnar): Add a label filter
            metric = SegmentationMAP(
                self.annotation_filters,
                self.prediction_filters,
                self.iou_thresholds,
            )
        else:

            def patched_average_precision(annotations, predictions):
                ap_per_threshold = []
                labels = [p.label for p in predictions.polygon_predictions]
                for threshold in self.iou_thresholds:
                    ap_per_label = []
                    for label in labels:
                        call_metric = PolygonAveragePrecision(
                            label,
                            iou_threshold=threshold,
                            annotation_filters=self.annotation_filters,
                            prediction_filters=self.prediction_filters,
                        )
                        result = call_metric(annotations, predictions)
                        ap_per_label.append(result.value)  # type: ignore
                    ap_per_threshold = np.mean(ap_per_label)

                thresholds = np.concatenate([[0], self.iou_thresholds, [1]])
                steps = np.diff(thresholds)
                mean_ap = (
                    np.array(ap_per_threshold + [ap_per_threshold[-1]]) * steps
                ).sum()
                return ScalarResult(mean_ap)

            metric = patched_average_precision
        return metric
