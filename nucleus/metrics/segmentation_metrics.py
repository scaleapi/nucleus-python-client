import abc
from typing import List, Optional, Set, Tuple, Union

import numpy as np

from nucleus.annotation import AnnotationList, Segment, SegmentationAnnotation
from nucleus.metrics.base import MetricResult
from nucleus.metrics.filtering import ListOfAndFilters, ListOfOrAndFilters
from nucleus.prediction import PredictionList, SegmentationPrediction

from .base import Metric, ScalarResult
from .segmentation_loader import SegmentationMaskLoader
from .segmentation_utils import (
    FALSE_POSITIVES,
    convert_to_instance_seg_confusion,
    fast_confusion_matrix,
    non_max_suppress_confusion,
    setup_iou_thresholds,
)

try:
    from s3fs import S3FileSystem
except (ModuleNotFoundError, OSError):
    from ..package_not_installed import PackageNotInstalled

    S3FileSystem = PackageNotInstalled


# pylint: disable=useless-super-delegation


class SegmentationMaskMetric(Metric):
    def __init__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        iou_threshold: float = 0.5,
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
        self.iou_threshold = iou_threshold

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
        if len(img.shape) > 2:
            # TODO: Do we have to do anything more advanced? Currently expect all channels to have same data
            min_dim = np.argmin(img.shape)
            if min_dim == 0:
                img = img[0, :, :]
            elif min_dim == 1:
                img = img[:, 0, :]
            else:
                img = img[:, :, 0]
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
        self,
        annotation,
        annotation_img,
        prediction,
        prediction_img,
        iou_threshold,
    ) -> Tuple[np.ndarray, Set[int]]:
        """This calculates a confusion matrix with ground_truth_index X predicted_index summary

        Notes:
            If filtering has been applied we filter out missing segments from the confusion matrix.

        Returns:
            Class-based confusion matrix and a set of indexes that are not considered a part of the taxonomy (and are
            only considered for FPs not as a part of mean calculations)


        TODO(gunnar): Allow pre-seeding confusion matrix (all of the metrics calculate the same confusion matrix ->
            we can calculate it once and then use it for all other metrics in the chain)
        """
        # NOTE: This creates a max(class_index) * max(class_index) MAT. If we have np.int32 this could become
        #  huge. We could probably use a sparse matrix instead or change the logic to only create count(index) ** 2
        #  matrix (we only need to keep track of available indexes)
        num_classes = (
            max(
                max((a.index for a in annotation.annotations)),
                max((a.index for a in prediction.annotations)),
            )
            + 1  # to include 0
        )
        confusion = fast_confusion_matrix(
            annotation_img, prediction_img, num_classes
        )
        confusion = self._filter_confusion_matrix(
            confusion, annotation, prediction
        )
        confusion = non_max_suppress_confusion(confusion, iou_threshold)
        false_positive = Segment(FALSE_POSITIVES, index=confusion.shape[0] - 1)
        if annotation.annotations[-1].label != FALSE_POSITIVES:
            annotation.annotations.append(false_positive)
            if annotation.annotations is not prediction.annotations:
                # Probably likely that this structure is re-used -> check if same list instance and only append once
                # TODO(gunnar): Should this uniqueness be handled by the base class?
                prediction.annotations.append(false_positive)

        # TODO(gunnar): Detect non_taxonomy classes for segmentation as well as instance segmentation
        non_taxonomy_classes = set()
        if self._is_instance_segmentation(annotation, prediction):
            (
                confusion,
                _,
                non_taxonomy_classes,
            ) = convert_to_instance_seg_confusion(
                confusion, annotation, prediction
            )
        else:
            ann_labels = list(
                dict.fromkeys(s.label for s in annotation.annotations)
            )
            pred_labels = list(
                dict.fromkeys(s.label for s in prediction.annotations)
            )
            missing_or_filtered_labels = set(ann_labels) - set(pred_labels)
            non_taxonomy_classes = {
                segment.index
                for segment in annotation.annotations
                if segment.label in missing_or_filtered_labels
            }

        return confusion, non_taxonomy_classes

    def _is_instance_segmentation(self, annotation, prediction):
        """Guesses that we're dealing with instance segmentation if we have multiple segments with same label.
        Degenerate case is same as semseg so fine to misclassify in that case."""
        # This is a trick to get ordered sets
        ann_labels = list(
            dict.fromkeys(s.label for s in annotation.annotations)
        )
        pred_labels = list(
            dict.fromkeys(s.label for s in prediction.annotations)
        )
        # NOTE: We assume instance segmentation if labels are duplicated in annotations or predictions
        is_instance_segmentation = len(ann_labels) != len(
            annotation.annotations
        ) or len(pred_labels) != len(prediction.annotations)
        return is_instance_segmentation

    def _filter_confusion_matrix(self, confusion, annotation, prediction):
        if self.annotation_filters or self.prediction_filters:
            new_confusion = np.zeros_like(confusion)
            # we mask the confusion matrix instead of the images
            if self.annotation_filters:
                annotation_indexes = {
                    segment.index for segment in annotation.annotations
                }
                for row in annotation_indexes:
                    new_confusion[row, :] = confusion[row, :]
            if self.prediction_filters:
                prediction_indexes = {
                    segment.index for segment in prediction.annotations
                }
                for col in prediction_indexes:
                    new_confusion[:, col] = confusion[:, col]
            confusion = new_confusion
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
        iou_threshold: float = 0.5,
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
            iou_threshold,
        )

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        confusion, non_taxonomy_classes = self._calculate_confusion_matrix(
            annotation,
            annotation_img,
            prediction,
            prediction_img,
            self.iou_threshold,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            tp = confusion[:-1, :-1]
            fp = confusion[:, -1]
            iou = np.diag(tp) / (
                tp.sum(axis=1) + tp.sum(axis=0) + fp.sum() - np.diag(tp)
            )
            non_taxonomy_classes = non_taxonomy_classes - {
                confusion.shape[1] - 1
            }
            iou.put(list(non_taxonomy_classes), np.nan)
            mean_iou = np.nanmean(iou)
            return ScalarResult(value=mean_iou, weight=annotation_img.size)  # type: ignore

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
        iou_threshold: float = 0.5,
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
            iou_threshold,
        )

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        confusion, non_taxonomy_classes = self._calculate_confusion_matrix(
            annotation,
            annotation_img,
            prediction,
            prediction_img,
            self.iou_threshold,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # TODO(gunnar): Logic can be simplified
            confused = confusion[:-1, :-1]
            tp = confused.diagonal()
            fp = confusion[:, -1][:-1] + confused.sum(axis=0) - tp
            tp_and_fp = tp + fp
            precision = tp / tp_and_fp
            non_taxonomy_classes = non_taxonomy_classes - {
                confusion.shape[1] - 1
            }
            precision.put(list(non_taxonomy_classes), np.nan)
            avg_precision = np.nanmean(precision)
        return ScalarResult(value=np.nan_to_num(avg_precision), weight=confusion.sum())  # type: ignore

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
        iou_threshold: float = 0.5,
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
            annotation_filters,
            prediction_filters,
            iou_threshold,
        )

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        confusion, non_taxonomy_classes = self._calculate_confusion_matrix(
            annotation,
            annotation_img,
            prediction,
            prediction_img,
            self.iou_threshold,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            recall = confusion.diagonal() / confusion.sum(axis=1)
            recall.put(
                list(non_taxonomy_classes), np.nan
            )  # We don't consider non taxonomy classes, i.e. FPs and background
            mean_recall = np.nanmean(recall)
        return ScalarResult(value=np.nan_to_num(mean_recall), weight=annotation_img.size)  # type: ignore

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

    iou_setups = {"coco"}

    # TODO: Remove defaults once these are surfaced more cleanly to users.
    def __init__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        iou_thresholds: Union[List[float], str] = "coco",
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
            map_thresholds: Provide a list of threshold to compute over or literal "coco"
        """
        super().__init__(
            annotation_filters,
            prediction_filters,
        )
        self.iou_thresholds = setup_iou_thresholds(iou_thresholds)

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:

        ap_per_threshold = []
        weight = 0
        for iou_threshold in self.iou_thresholds:
            ap = SegmentationPrecision(
                self.annotation_filters, self.prediction_filters, iou_threshold
            )
            ap.loader = self.loader
            ap_result = ap(
                AnnotationList(segmentation_annotations=[annotation]),
                PredictionList(segmentation_predictions=[prediction]),
            )
            ap_per_threshold.append(ap_result.value)  # type: ignore
            weight += ap_result.weight  # type: ignore

        thresholds = np.concatenate([[0], self.iou_thresholds, [1]])
        steps = np.diff(thresholds)
        mean_ap = (
            np.array(ap_per_threshold + [ap_per_threshold[-1]]) * steps
        ).sum()
        return ScalarResult(mean_ap, weight=weight)

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
        iou_threshold: float = 0.5,
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
            annotation_filters,
            prediction_filters,
            iou_threshold,
        )

    def _metric_impl(
        self,
        annotation_img: np.ndarray,
        prediction_img: np.ndarray,
        annotation: SegmentationAnnotation,
        prediction: SegmentationPrediction,
    ) -> ScalarResult:
        confusion, non_taxonomy_classes = self._calculate_confusion_matrix(
            annotation,
            annotation_img,
            prediction,
            prediction_img,
            self.iou_threshold,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(confusion) / (
                confusion.sum(axis=1)
                + confusion.sum(axis=0)
                - np.diag(confusion)
            )
            predicted_counts = confusion.sum(axis=0).astype(np.float_)
            predicted_counts.put(list(non_taxonomy_classes), np.nan)
            freq = predicted_counts / np.nansum(predicted_counts)
            iu.put(list(non_taxonomy_classes), np.nan)
            fwavacc = (
                np.nan_to_num(freq[freq > 0]) * np.nan_to_num(iu[freq > 0])
            ).sum()
            mean_fwavacc = np.nanmean(fwavacc)
        return ScalarResult(value=np.nan_to_num(mean_fwavacc), weight=confusion.sum())  # type: ignore

    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        return ScalarResult.aggregate(results)  # type: ignore
