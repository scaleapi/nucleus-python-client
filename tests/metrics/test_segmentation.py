from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from nucleus.annotation import AnnotationList, Segment, SegmentationAnnotation
from nucleus.metrics import ScalarResult
from nucleus.metrics.filtering import SegmentFieldFilter
from nucleus.metrics.segmentation_loader import InMemoryLoader
from nucleus.metrics.segmentation_metrics import (
    SegmentationFWAVACC,
    SegmentationIOU,
    SegmentationMAP,
    SegmentationPrecision,
    SegmentationRecall,
)
from nucleus.metrics.segmentation_utils import (
    FALSE_POSITIVES,
    convert_to_instance_seg_confusion,
    fast_confusion_matrix,
    max_iou_match_from_confusion,
    non_max_suppress_confusion,
)
from nucleus.prediction import PredictionList, SegmentationPrediction


@dataclass
class SegmentationTestSetup:
    """Common segmentation test setup

    Attributes:
        gt: Ground truth image
        pred: predicted image
        expected_results: Expected aggregated result from the image
        annotations: Create a custom segmentation hierarchy (i.e. to test instance segmentation)
        iou_threshold: Set a custom iou_threshold. We default to 0 for easy mental calculations.
    """

    gt: List[int]
    pred: List[int]
    expected_result: float
    gt_annotations: Optional[List[Segment]] = None
    pred_annotations: Optional[List[Segment]] = None
    iou_threshold: float = 0


def compose_input_variables(setup: SegmentationTestSetup):
    """Common step to create input variables from SegmentationTestSetup"""
    gt_annotations = (
        setup.gt_annotations
        if setup.gt_annotations
        else [
            Segment(f"{index}", index) for index in set(setup.gt + setup.pred)
        ]
    )
    pred_annotations = (
        setup.pred_annotations if setup.pred_annotations else gt_annotations
    )
    annotation = SegmentationAnnotation(
        "s3://fake_ann_url",
        annotations=gt_annotations,
        reference_id="item_1",
    )
    prediction = SegmentationPrediction(
        "s3://fake_pred_url",
        annotations=pred_annotations,
        reference_id=annotation.reference_id,
    )
    ground_truth_img = np.array(setup.gt, dtype=np.int32)
    prediction_img = np.array(setup.pred, dtype=np.int32)
    url_to_img = {
        annotation.mask_url: ground_truth_img,
        prediction.mask_url: prediction_img,
    }
    return annotation, prediction, url_to_img


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup([1], [0], 0),
        SegmentationTestSetup([1], [1], 1),
        SegmentationTestSetup([1, 1], [0, 1], 0.25),
        SegmentationTestSetup([1, 1, 1, 1], [0, 1, 1, 1], 0.75 / 2),
        SegmentationTestSetup([0, 1, 2, 3], [0, 0, 0, 0], (1 / 4) / 4),
    ],
)
def test_segmentation_iou(setup):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationIOU()
    metric.loader = InMemoryLoader(url_to_img)
    result = metric(
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == setup.expected_result


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup([1], [0], 0),
        SegmentationTestSetup([1], [1], 1),
        SegmentationTestSetup([1, 1], [0, 1], 0.5),
        SegmentationTestSetup([1, 1, 1, 1], [0, 1, 1, 1], 0.75),
        SegmentationTestSetup([0, 1, 2, 3], [0, 0, 0, 0], 0.25),
    ],
)
def test_segmentation_recall(setup):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationRecall(iou_threshold=setup.iou_threshold)
    metric.loader = InMemoryLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == setup.expected_result


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup([1], [0], 0),
        SegmentationTestSetup([1], [1], 1),
        SegmentationTestSetup([1, 1], [0, 1], 0.5),
        SegmentationTestSetup([1, 1, 1, 1], [0, 1, 1, 1], (0 + 1) / 2),
        SegmentationTestSetup([0, 1, 2, 3], [0, 0, 0, 0], (1 + 0 + 0 + 0) / 4),
    ],
)
def test_segmentation_precision(setup):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationPrecision(iou_threshold=setup.iou_threshold)
    metric.loader = InMemoryLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == setup.expected_result


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup([1], [0], 0),
        SegmentationTestSetup([1], [1], 1),
        SegmentationTestSetup([1, 1], [0, 1], (1 * 0 + 1 * 0.5) / 2),
        SegmentationTestSetup(
            [1, 1, 1, 1], [0, 1, 1, 1], (0.25 * 0 + 0.75 * 3 / 4)
        ),
        SegmentationTestSetup(
            [0, 1, 1, 3], [0, 1, 1, 0], (1 * 1 + 2 * 1 + 1 * 0) / 4
        ),
    ],
)
def test_segmentation_fwavacc(setup):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationFWAVACC(iou_threshold=setup.iou_threshold)
    metric.loader = InMemoryLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == setup.expected_result


@pytest.mark.parametrize(
    "setup, iou_thresholds",
    [
        (SegmentationTestSetup([1], [0], 0), "coco"),
        (SegmentationTestSetup([1], [1], 1), "coco"),
        (
            SegmentationTestSetup(
                [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                ((1 / 1 + 3 / 4 + 5 / 6) / 3 + (0 + 0 + 5 / 6) / 3) / 2,
            ),
            [0.5, 0.8],
        ),
    ],
)
def test_segmentation_map(setup, iou_thresholds):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationMAP(iou_thresholds=iou_thresholds)
    metric.loader = InMemoryLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert_almost_equal(result.value, setup.expected_result)


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
    metric.loader = InMemoryLoader(url_to_img)
    result: ScalarResult = metric(  # type: ignore
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert result.value == 1.0


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup(
            [1, 2],
            [0, 0],
            0,
            [
                Segment("background", 0),
                Segment("instance", 1),
                Segment("instance", 2),
            ],
        ),
        SegmentationTestSetup(
            [1, 2],
            [2, 1],
            1,
            [
                Segment("background", 0),
                Segment("instance", 1),
                Segment("instance", 2),
            ],
        ),
        SegmentationTestSetup(
            [0, 0, 1, 2],
            [0, 1, 2, 1],
            0.75,
            [
                Segment("background", 0),
                Segment("class1", 1),
                Segment("class1", 2),
                Segment("class2", 3),
            ],
        ),
        SegmentationTestSetup(
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            0.0,  # non-max suppression causes more FPs to match
            [
                Segment("background", 0),
                Segment("class1", 1),
                Segment("class1", 2),
                Segment("class2", 3),
            ],
        ),
        SegmentationTestSetup(
            [0, 0, 0, 0, 1, 1, 1],
            [0, 1, 2, 1, 1, 1, 2],
            (0 / 4 + 2 / 3) / 2,  # class 0 recall + class 1 recall averaged
            [
                Segment("background", 0),
                Segment("class1", 1),
                Segment("class1", 2),
            ],
        ),
    ],
)
def test_instance_segmentation_recall(setup: SegmentationTestSetup):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    # iou_threshold=0 is easier to reason about for small "images"
    metric = SegmentationRecall(iou_threshold=setup.iou_threshold)
    metric.loader = InMemoryLoader(url_to_img)
    result = metric(
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert_almost_equal(result.value, setup.expected_result)


@pytest.mark.parametrize(
    "setup",
    [
        SegmentationTestSetup(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            (3 / 5 + 5 / 5) / 2,  # class 0 recall + class 1 recall averaged
            [
                Segment("background", 0),
                Segment("class1", 1),
                Segment("class1", 2),
            ],
        ),
        SegmentationTestSetup(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            (0 / 5 + 5 / 5)
            / 2,  # Note the threshold below. expect 0 recall since IoU<0.75
            [
                Segment("background", 0),
                Segment("class1", 1),
                Segment("class1", 2),
            ],
            iou_threshold=0.75,
        ),
    ],
)
def test_instance_segmentation_recall_varying_thresholds(
    setup: SegmentationTestSetup,
):
    annotation, prediction, url_to_img = compose_input_variables(setup)

    metric = SegmentationRecall(iou_threshold=setup.iou_threshold)
    metric.loader = InMemoryLoader(url_to_img)
    result = metric(
        AnnotationList(segmentation_annotations=[annotation]),
        PredictionList(segmentation_predictions=[prediction]),
    )
    assert_almost_equal(result.value, setup.expected_result)


@dataclass
class IOUMatch:
    gt: int
    pred: int
    iou: float


@pytest.mark.parametrize(
    "gt,pred,expected_matches",
    [
        (
            [0],
            [1],
            [
                IOUMatch(gt=0, pred=0, iou=0),
                IOUMatch(gt=1, pred=1, iou=0),
            ],
        ),
        (
            [0],
            [0],
            [
                IOUMatch(gt=0, pred=0, iou=1),
            ],
        ),
        (
            [0, 0, 0, 0, 1, 1, 1],
            [0, 1, 2, 1, 1, 1, 2],
            [
                IOUMatch(gt=0, pred=0, iou=1 / 4),  # 1 pred / 4 gt ...
                IOUMatch(gt=1, pred=1, iou=2 / 5),
                IOUMatch(gt=2, pred=2, iou=0),
            ],
        ),
    ],
)
def test_iou_match_confusion_matrix(
    gt, pred, expected_matches: List[IOUMatch]
):
    ground_truth_img = np.array(gt, dtype=np.int32)
    prediction_img = np.array(pred, dtype=np.int32)
    n_class = max(max(gt), max(pred)) + 1
    confusion = fast_confusion_matrix(
        ground_truth_img, prediction_img, n_class=n_class
    )
    iou, max_iou_rows, max_iou_cols = max_iou_match_from_confusion(confusion)

    for idx, expected_match in enumerate(expected_matches):
        matched_row = max_iou_rows[idx]
        matched_col = max_iou_cols[idx]
        matched_iou = iou[matched_row, matched_col]
        if expected_match.iou > 0:
            assert_almost_equal(matched_iou, expected_match.iou)
            assert matched_row == expected_match.gt
            assert matched_col == expected_match.pred


@pytest.mark.parametrize(
    "setup,expected_confusion,expected_non_max_confusion,expected_class_confusion",
    [
        # (
        #     SegmentationTestSetup(
        #         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        #         [2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3],
        #         0,
        #         gt_annotations=[
        #             Segment("background", 0),
        #             Segment("class1", 1),
        #             Segment("class1", 2),
        #             Segment("class1", 3),
        #         ],
        #         pred_annotations=[
        #             Segment("background", 0),
        #             Segment("class1", 3),
        #             Segment("class1", 2),
        #             Segment("class1", 1),
        #         ],
        #     ),
        #     np.array([[3, 1, 1, 0], [0, 3, 2, 0], [0, 0, 0, 2], [0, 0, 0, 0]]),
        #     np.array(
        #         [
        #             [3, 1, 1, 0, 0],  # TODO(gunnar):
        #             [0, 3, 2, 0, 0],  # Maybe should be [0, 3, 0, 0, 2],
        #             [0, 0, 0, 2, 0],  # since we're doing instance seg
        #             [0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0],
        #         ]
        #     ),
        #     np.array([[3, 2, 0], [0, 5, 2], [0, 0, 0]]),
        # ),
        (
            SegmentationTestSetup(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
                [2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3],
                0,
                gt_annotations=[
                    Segment("class0", 0),
                    Segment("class1", 1),
                    Segment("class2", 2),
                    Segment("class1", 3),
                ],
                pred_annotations=[
                    Segment("class0", 0),
                    Segment("class1", 3),
                    Segment("class1", 2),
                    Segment("class2", 1),
                ],
            ),
            np.array([[3, 1, 1, 0], [0, 3, 2, 0], [0, 0, 0, 2], [0, 0, 0, 0]]),
            np.array(
                [
                    [3, 1, 1, 0, 0],  # TODO(gunnar):
                    [0, 3, 2, 0, 0],  # Maybe should be [0, 3, 0, 0, 2],
                    [0, 0, 0, 2, 0],  # since we're doing instance seg
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array([[3, 1, 1, 0], [0, 0, 3, 2], [0, 2, 0, 0], [0, 0, 0, 0]]),
        ),
    ],
)
def test_instance_confusion_chain(
    setup,
    expected_confusion,
    expected_non_max_confusion,
    expected_class_confusion,
):
    num_classes = max(max(setup.gt), max(setup.pred)) + 1
    actual_confusion = fast_confusion_matrix(
        np.array(setup.gt), np.array(setup.pred), num_classes
    )
    annotation, prediction, url_to_img = compose_input_variables(setup)
    assert_array_equal(actual_confusion, expected_confusion)
    nms_confusion = non_max_suppress_confusion(
        actual_confusion, setup.iou_threshold
    )
    false_positive = Segment(
        FALSE_POSITIVES, index=actual_confusion.shape[0] - 1
    )
    annotation.annotations.append(false_positive)
    if annotation.annotations is not prediction.annotations:
        prediction.annotations.append(false_positive)
    assert_array_equal(nms_confusion, expected_non_max_confusion)
    (
        actual_class_confusion,
        new_labels,
        non_taxonomy_classes,
    ) = convert_to_instance_seg_confusion(
        nms_confusion, annotation, prediction
    )
    assert new_labels == ["class0", "class1", "class2", FALSE_POSITIVES]
    assert non_taxonomy_classes == {3}
    assert_array_equal(actual_class_confusion, expected_class_confusion)
