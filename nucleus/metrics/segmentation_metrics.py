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

# pylint: disable=useless-super-delegation


def _fast_hist(
    label_true: np.ndarray, label_pred: np.ndarray, n_class: int
) -> np.ndarray:
    """Calculates confusion matrix - fast!

    Outputs a confusion matrix where each row is GT confusion and column is prediction confusion
    Example:
        _fast_hist(np.array([0, 1, 2, 3], dtype=np.int16), np.array([0, 1, 1, 1], dtype=np.int16), n_class=4)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0],
               [0, 1, 0, 0]])
    """
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
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonMetric abstract object.

        Args:
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
        # TODO -> add custom filtering to Segmentation(Annotation|Prediction).annotations.(metadata|label)
        super().__init__(annotation_filters, prediction_filters)
        self.loader = SegmentationMaskLoader(S3FileSystem(anon=False))

    def call_metric(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> ScalarResult:
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
        if (
            annotation
            and prediction
            and annotation.annotations
            and prediction.annotations
        ):
            annotation_img = self.get_mask_channel(annotation)
            pred_img = self.get_mask_channel(prediction)
            return self._metric_impl(
                np.asarray(annotation_img, dtype=np.int32),
                np.asarray(pred_img, dtype=np.int32),
                annotation,
                prediction,
            )
        else:
            return ScalarResult(0, weight=0)

    def get_mask_channel(self, ann_or_pred):
        """Some annotations are stored as RGB instead of L (single-channel).
        We expect the image to be faux-single-channel with all the channels repeating so we choose the first one.
        """
        img = self.loader.fetch(ann_or_pred.mask_url)
        if len(img.getbands()) > 1:
            # TODO: Do we have to do anything more advanced? Currently expect all channels to have same data
            img = img.getchannel(0)
        return img

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
    ) -> np.ndarray:
        """This calculates a confusion matrix with ground_truth_index X predicted_index summary

        Notes:
            If filtering has been applied we filter out missing segments from the confusion matrix.

        TODO(gunnar): Allow pre-seeding confusion matrix (all of the metrics calculate the same confusion matrix ->
            we can calculate it once and then use it for all other metrics in the chain)
        """
        # NOTE: This creates a max(class_index) * max(class_index) MAT. If we have np.int16 this could become
        #  huge. We could probably use a sparse matrix instead or change the logic to only create count(index) ** 2
        #  matrix (we only need to keep track of available indexes)
        num_classes = (
            max(
                max((a.index for a in annotation.annotations)),
                max((a.index for a in prediction.annotations)),
            )
            + 1  # to include 0
        )
        confusion = _fast_hist(annotation_img, prediction_img, num_classes)
        if self.annotation_filters or self.prediction_filters:
            # we mask the confusion matrix instead of the images
            if self.annotation_filters:
                annotation_indexes = {
                    segment.index for segment in annotation.annotations
                }
                indexes_to_remove = (
                    set(range(confusion.shape[0] - 1)) - annotation_indexes
                )
                for row in indexes_to_remove:
                    confusion[row, :] = 0
            if self.prediction_filters:
                prediction_indexes = {
                    segment.index for segment in prediction.annotations
                }
                indexes_to_remove = (
                    set(range(confusion.shape[0] - 1)) - prediction_indexes
                )
                for col in indexes_to_remove:
                    confusion[:, col] = 0
        return confusion


class SegmentationIOU(SegmentationMaskMetric):
    def __init__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonIOU object.

        Args:
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
        super().__init__(
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
        confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.diag(confusion) / (
                confusion.sum(axis=1)
                + confusion.sum(axis=0)
                - np.diag(confusion)
            )
        return ScalarResult(value=np.nanmean(iou), weight=annotation_img.size)  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return ScalarResult.aggregate(results)  # type: ignore


class SegmentationPrecision(SegmentationMaskMetric):
    def __init__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Calculates mean per-class precision

        Args:
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
        super().__init__(
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
        confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(confusion)
            precision = np.nanmean(true_pos / np.sum(confusion, axis=1))
        return ScalarResult(value=precision, weight=confusion.sum())  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return ScalarResult.aggregate(results)  # type: ignore


class SegmentationRecall(SegmentationMaskMetric):
    """Calculates the recall for a segmentation mask"""

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonRecall object.

        Args:
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
        super().__init__(
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

        confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            true_pos = np.diag(confusion)
            recall = np.nanmean(true_pos / np.sum(confusion, axis=1))
        return ScalarResult(value=recall, weight=annotation_img.size)  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return ScalarResult.aggregate(results)  # type: ignore


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
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes PolygonRecall object.

        Args:
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
        super().__init__(
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

        confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )
        label_to_index = {a.label: a.index for a in annotation.annotations}
        num_classes = len(label_to_index.keys())
        ap_per_class = np.ndarray(num_classes)  # type: ignore
        with np.errstate(divide="ignore", invalid="ignore"):
            for class_idx, (_, index) in enumerate(label_to_index.items()):
                true_pos = confusion[index, index]
                false_pos = confusion[:, index].sum()
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
        return ScalarResult.aggregate(results)  # type: ignore


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
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
    ):
        """Initializes SegmentationFWAVACC object.

        Args:
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
        super().__init__(
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

        confusion = self._calculate_confusion_matrix(
            annotation, annotation_img, prediction, prediction_img
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(confusion) / (
                confusion.sum(axis=1)
                + confusion.sum(axis=0)
                - np.diag(confusion)
            )
            freq = confusion.sum(axis=1) / confusion.sum()
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return ScalarResult(value=np.nanmean(fwavacc), weight=1)  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return ScalarResult.aggregate(results)  # type: ignore
