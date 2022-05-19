import abc
from typing import List, Optional, Union

import fsspec
import numpy as np
from PIL import Image
from s3fs import S3FileSystem

from nucleus.annotation import AnnotationList, SegmentationAnnotation
from nucleus.metrics.base import MetricResult
from nucleus.metrics.filtering import ListOfAndFilters, ListOfOrAndFilters
from nucleus.prediction import PredictionList, SegmentationPrediction

from .base import Metric, ScalarResult
from .metric_utils import compute_average_precision


def _fast_hist(label_true, label_pred, n_class):
    """Calculates confusion matrix - fast!"""
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


class SegmentationMaskLoader:
    def __init__(self, fs: fsspec):
        self.fs = fs

    def fetch(self, url: str):
        with self.fs.open(url) as fh:
            img = Image.open(fh)
        return img


class SegmentationMaskMetric(Metric):
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
        # NOTE: We store histogram for re-use in subsequently calculated metrics
        self.confusion: Optional[np.ndarray] = None

    def call_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> MetricResult:
        assert (
            len(annotations.segmentation_annotations) <= 1
        ), f"Expected only one segmentation mask, got {annotations.segmentation_annotations}"
        assert (
            len(predictions.segmentation_predictions) <= 1
        ), f"Expected only one segmentation mask, got {predictions.segmentation_predictions}"
        annotation = (
            annotations.segmentation_annotations[0]
            if annotations.segmentation_annotations
            else None
        )
        prediction = (
            predictions.segmentation_predictions[0]
            if predictions.segmentation_predictions
            else None
        )
        annotation_img = self.loader.fetch(annotation.mask_url)
        pred_img = self.loader.fetch(prediction.mask_url)
        return self._metric_impl(
            np.asarray(annotation_img, dtype=np.int32),
            np.asarray(pred_img, dtype=np.int32),
            annotation,
            prediction,
        )

    @abc.abstractmethod
    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ):
        pass

    def _calculate_confusion_matrix(
        self, annotation, annotation_img, prediction, prediction_img
    ):
        num_classes = (
            max(
                max((a.index for a in annotation.annotations)),
                max((a.index for a in prediction.annotations)),
            )
            + 1
        )
        confusion = (
            _fast_hist(annotation_img, prediction_img, num_classes)
            if self.confusion is None
            else self.confusion
        )
        return confusion


class SegmentationIOU(SegmentationMaskMetric):
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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(self.confusion)
            false_pos = np.sum(self.confusion, axis=0) - true_pos
            false_neg = np.sum(self.confusion, axis=1) - true_pos
            mean_iou = true_pos / (false_neg + false_pos - true_pos)
        return ScalarResult(
            value=np.nanmean(mean_iou), weight=annotation_img.size
        )

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationPrecision(SegmentationMaskMetric):
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
        """Calculates mean per-class precision

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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(self.confusion)
            precision = true_pos / np.sum(self.confusion, axis=0)
            mean_precision = np.nanmean(precision)
        return ScalarResult(value=mean_precision, weight=1)  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationAveragePrecision(SegmentationMaskMetric):
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
        """Initializes SegmentationPrecision object.

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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(self.confusion)
            precision = true_pos / np.sum(self.confusion, axis=0)
            recall = true_pos / np.sum(self.confusion, axis=1)
            average_precision = compute_average_precision(
                np.nan_to_num(recall), np.nan_to_num(precision)
            )
        return ScalarResult(value=average_precision, weight=1)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationAverageRecall(SegmentationMaskMetric):
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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(self.confusion)
            recall = np.nanmean(true_pos / np.sum(self.confusion, axis=1))
        return ScalarResult(value=recall, weight=annotation_img.size)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationMAP(SegmentationMaskMetric):
    """Calculates the mean average precision per class for segmentation masks
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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )
        label_to_index = {a.label: a.index for a in annotation.annotations}
        num_classes = len(label_to_index.keys())
        ap_per_class = np.ndarray((num_classes))
        with np.errstate(divide="ignore", invalid="ignore"):
            for class_idx, (_, index) in enumerate(label_to_index.items()):
                true_pos = self.confusion[index, index]
                false_pos = self.confusion[:, index].sum()
                samples = true_pos + false_pos
                if samples:
                    ap_per_class[class_idx] = true_pos / samples
                else:
                    ap_per_class[class_idx] = np.nan

        if num_classes > 0:
            m_ap = np.nanmean(ap_per_class)
            return ScalarResult(m_ap, weight=1)  # type: ignore
        else:
            return ScalarResult(0, weight=0)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore


class SegmentationFWAVACC(SegmentationMaskMetric):
    """Calculates the frequency weighted average of the class-wise Jaccard index
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

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        self.confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(self.confusion) / (
                self.confusion.sum(axis=1)
                + self.confusion.sum(axis=0)
                - np.diag(self.confusion)
            )
            freq = self.confusion.sum(axis=1) / self.confusion.sum()
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return ScalarResult(value=np.nanmean(fwavacc), weight=1)

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return self.metric.aggregate_score(results)  # type: ignore
