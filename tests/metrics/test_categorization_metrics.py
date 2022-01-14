from copy import deepcopy

import pytest

from nucleus.metrics import PolygonIOU, PolygonPrecision, PolygonRecall
from nucleus.metrics.base import Metric, ScalarResult
from tests.metrics.helpers import (
    TEST_ANNOTATION_LIST,
    TEST_BOX_ANNOTATION_LIST,
    TEST_BOX_PREDICTION_LIST,
    TEST_CONVEX_POLYGON_ANNOTATION_LIST,
    TEST_CONVEX_POLYGON_PREDICTION_LIST,
    TEST_PREDICTION_LIST,
    assert_metric_eq,
)


def test_perfect_match_polygon_metrics(
    test_annotations, test_predictions, metric_fn
):
    # Test metrics on where annotations = predictions perfectly
    metric = metric_fn(enforce_label_match=False)
    result = metric(test_annotations, test_predictions)
    assert_metric_eq(result, ScalarResult(1, len(test_annotations)))

    metric = metric_fn(enforce_label_match=True)
    result = metric(test_annotations, test_predictions)
    assert_metric_eq(result, ScalarResult(1, len(test_annotations)))


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn",
    [
        (TEST_BOX_ANNOTATION_LIST, TEST_BOX_PREDICTION_LIST, PolygonIOU),
        (TEST_BOX_ANNOTATION_LIST, TEST_BOX_PREDICTION_LIST, PolygonPrecision),
        (TEST_BOX_ANNOTATION_LIST, TEST_BOX_PREDICTION_LIST, PolygonRecall),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonIOU,
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonPrecision,
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonRecall,
        ),
    ],
)
def test_perfect_unmatched_polygon_metrics(
    test_annotations, test_predictions, metric_fn
):
    # Test metrics on where annotations and predictions do not have matching reference IDs.
    test_predictions_unmatch = deepcopy(test_predictions)
    for box in test_predictions_unmatch.box_predictions:
        box.reference_id += "_bad"
    for polygon in test_predictions_unmatch.polygon_predictions:
        polygon.reference_id += "_bad"
    metric = metric_fn(enforce_label_match=False)
    result = metric(test_annotations, test_predictions_unmatch)
    assert_metric_eq(result, ScalarResult(0, len(test_annotations)))

    metric = metric_fn(enforce_label_match=True)
    result = metric(test_annotations, test_predictions_unmatch)
    assert_metric_eq(result, ScalarResult(0, len(test_annotations)))


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn,expected",
    [
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonIOU,
            ScalarResult(0.545, 2),
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonPrecision,
            ScalarResult(0.5, 2),
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonRecall,
            ScalarResult(0.5, 2),
        ),
    ],
)
def test_simple_2_boxes(
    test_annotations, test_predictions, metric_fn, expected
):
    # Test metrics on where annotations = predictions perfectly
    metric = metric_fn()
    result = metric(test_annotations, test_predictions)
    assert_metric_eq(result, expected)
