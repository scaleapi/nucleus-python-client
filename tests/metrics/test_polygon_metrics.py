from copy import deepcopy

import pytest

from nucleus.metrics import (
    PolygonAveragePrecision,
    PolygonIOU,
    PolygonMAP,
    PolygonPrecision,
    PolygonRecall,
)
from nucleus.metrics.base import MetricResult
from tests.metrics.helpers import (
    TEST_ANNOTATION_LIST,
    TEST_BOX_ANNOTATION_LIST,
    TEST_BOX_PREDICTION_LIST,
    TEST_CONVEX_POLYGON_ANNOTATION_LIST,
    TEST_CONVEX_POLYGON_PREDICTION_LIST,
    TEST_PREDICTION_LIST,
    assert_metric_eq,
)


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn,kwargs",
    [
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": False},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": False},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": False},
        ),
        (TEST_BOX_ANNOTATION_LIST, TEST_BOX_PREDICTION_LIST, PolygonMAP, {}),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonMAP,
            {},
        ),
    ],
)
def test_perfect_match_polygon_metrics(
    test_annotations, test_predictions, metric_fn, kwargs
):
    # Test metrics on where annotations = predictions perfectly
    metric = metric_fn(**kwargs)
    result = metric(test_annotations, test_predictions)
    assert_metric_eq(result, MetricResult(1, len(test_annotations)))


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn,kwargs",
    [
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": False},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": False},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": False},
        ),
        (TEST_BOX_ANNOTATION_LIST, TEST_BOX_PREDICTION_LIST, PolygonMAP, {}),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonIOU,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonPrecision,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": False},
        ),
        (
            TEST_CONVEX_POLYGON_ANNOTATION_LIST,
            TEST_CONVEX_POLYGON_PREDICTION_LIST,
            PolygonMAP,
            {},
        ),
    ],
)
def test_perfect_unmatched_polygon_metrics(
    test_annotations, test_predictions, metric_fn, kwargs
):
    # Test metrics on where annotations and predictions do not have matching reference IDs.
    test_predictions_unmatch = deepcopy(test_predictions)
    for box in test_predictions_unmatch.box_predictions:
        box.reference_id += "_bad"
    for polygon in test_predictions_unmatch.polygon_predictions:
        polygon.reference_id += "_bad"
    metric = metric_fn(**kwargs)
    result = metric(test_annotations, test_predictions_unmatch)
    assert_metric_eq(result, MetricResult(0, len(test_annotations)))


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn,expected,kwargs",
    [
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonIOU,
            MetricResult(109.0 / 300, 3),
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonIOU,
            MetricResult(109.0 / 300, 3),
            {"enforce_label_match": False},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonPrecision,
            MetricResult(1.0 / 3, 3),
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonPrecision,
            MetricResult(1.0 / 3, 3),
            {"enforce_label_match": False},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonRecall,
            MetricResult(0.5, 2),
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonRecall,
            MetricResult(0.5, 2),
            {"enforce_label_match": False},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonAveragePrecision,
            MetricResult(1.0 / 6, 1),
            {"label": "car"},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonMAP,
            MetricResult(1.0 / 6, 1),
            {},
        ),
    ],
)
def test_simple_2_boxes(
    test_annotations, test_predictions, metric_fn, expected, kwargs
):
    # Test metrics on where annotations = predictions perfectly
    metric = metric_fn(**kwargs)
    result = metric(test_annotations, test_predictions)
    assert_metric_eq(result, expected)
