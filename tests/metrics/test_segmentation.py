import numpy as np
import PIL.Image

from nucleus.annotation import AnnotationList, Segment, SegmentationAnnotation
from nucleus.metrics import ScalarResult
from nucleus.metrics.segmentation_metrics import (
    SegmentationAveragePrecision,
    SegmentationFWAVACC,
    SegmentationIOU,
    SegmentationMAP,
    SegmentationRecall,
    SegmentationPrecision,
)
from nucleus.prediction import PredictionList, SegmentationPrediction


class FakeLoader:
    def __init__(self, url_to_array):
        self.url_to_array = url_to_array

    def fetch(self, url: str):
        array = self.url_to_array[url]
        img = PIL.Image.fromarray(array)
        return img


def test_segmentation_iou_perfect():
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
    ground_truth_img = prediction_img = np.ones((5, 5))
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
    assert result.value == 1


def test_segmentation_iou_0():
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
    prediction_img = np.zeros((5, 5))
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationIOU()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 0


def test_segmentation_recall():
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
    prediction_img = np.ones((5, 5))
    prediction_img[:, 0] = 0
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationRecall()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 0.8


def test_segmentation_precision():
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
    prediction_img = np.ones((5, 5))
    prediction_img[:, 0:1] = 0
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
    # One class has precision 1.0, the other 0 -> 0.5
    assert result.value == 0.5


def test_segmentation_avg_precision():
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
    prediction_img = np.ones((5, 5))
    prediction_img[:, 0:1] = 0
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }

    metric = SegmentationAveragePrecision()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 0.8


def test_segmentation_fwavacc():
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

    metric = SegmentationFWAVACC()
    metric.loader = FakeLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 0.8


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
