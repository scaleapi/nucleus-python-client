from typing import Dict

import numpy as np
import PIL.Image
import pytest

from nucleus.annotation import AnnotationList, Segment, SegmentationAnnotation
from nucleus.metrics import ScalarResult
from nucleus.metrics.filtering import SegmentFieldFilter
from nucleus.metrics.segmentation_metrics import (
    SegmentationFWAVACC,
    SegmentationIOU,
    SegmentationMAP,
    SegmentationPrecision,
    SegmentationRecall,
)
from nucleus.prediction import PredictionList, SegmentationPrediction


class FakeLoader:
    """We use this loader in the tests, this allows us to serve images from memory instead of fetching
    from a filesystem.
    """

    def __init__(self, url_to_array: Dict[str, np.ndarray]):
        self.url_to_array = url_to_array

    def fetch(self, url: str):
        array = self.url_to_array[url]
        img = PIL.Image.fromarray(array)
        return img


@pytest.mark.parametrize(
    "gt,pred,iou",
    [
        ([1], [0], 0),
        ([1], [1], 1),
        ([1, 1], [0, 1], 0.25),
        ([1, 1, 1, 1], [0, 1, 1, 1], 0.75 / 2),
        ([0, 1, 2, 3], [0, 0, 0, 0], 0.25 / 4),
    ],
)
def test_segmentation_iou(gt, pred, iou):
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=[Segment(f"{index}", index) for index in set(gt + pred)],
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.array(gt, dtype=np.int16)
    prediction_img = np.array(pred, dtype=np.int16)
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationIOU()
    metric.loader = FakeLoader(url_to_img)
    result = metric(
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == iou


@pytest.mark.parametrize(
    "gt,pred,recall",
    [
        ([1], [0], 0),
        ([1], [1], 1),
        ([1, 1], [0, 1], 0.5),
        ([1, 1, 1, 1], [0, 1, 1, 1], 0.75),
        ([0, 1, 2, 3], [0, 0, 0, 0], 0.25),
    ],
)
def test_segmentation_recall(gt, pred, recall):
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=[Segment(f"{index}", index) for index in set(gt + pred)],
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.array(gt, dtype=np.int16)
    prediction_img = np.array(pred, dtype=np.int16)
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationRecall()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == recall


@pytest.mark.parametrize(
    "gt,pred,precision",
    [
        ([1], [0], 0),
        ([1], [1], 1),
        ([1, 1], [0, 1], 0.5),
        ([1, 1, 1, 1], [0, 1, 1, 1], 0.75),
        ([0, 1, 2, 3], [0, 0, 0, 0], 0.25),
    ],
)
def test_segmentation_precision(gt, pred, precision):
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=[Segment(f"{index}", index) for index in set(gt + pred)],
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.array(gt, dtype=np.int16)
    prediction_img = np.array(pred, dtype=np.int16)
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationPrecision()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == precision


@pytest.mark.parametrize(
    "gt,pred,fwavacc",
    [
        ([1], [0], 0),
        ([1], [1], 1),
        ([1, 1], [0, 1], 0.5),
        ([1, 1, 1, 1], [0, 1, 1, 1], 0.75),
        ([0, 1, 1, 3], [0, 1, 1, 0], 0.625),
    ],
)
def test_segmentation_fwavacc(gt, pred, fwavacc):
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=[Segment(f"{index}", index) for index in set(gt + pred)],
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.array(gt, dtype=np.int16)
    prediction_img = np.array(pred, dtype=np.int16)
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationFWAVACC()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == fwavacc


def test_segmentation_map():
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=[Segment("one", 1)],
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.ones((5, 5))
    prediction_img = np.copy(ground_truth_img)
    prediction_img[:, 0:1] = 0
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationMAP()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 0.5


def test_masked_recall():
    target_segment = Segment("one", 1)
    filter_segment = Segment("two", 2)
    segments = [target_segment, filter_segment]
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=segments,
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=annotation.annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.ones((5, 5))
    ground_truth_img[:, 0:1] = filter_segment.index
    prediction_img = np.ones((5, 5))
    prediction_img[:, 0:1] = 0
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    filters = [SegmentFieldFilter("index", "==", 1)]
    metric = SegmentationPrecision(
        annotation_filters=[], prediction_filters=filters
    )
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 1.0
