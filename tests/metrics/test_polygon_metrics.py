from copy import deepcopy

import pytest

from nucleus.metrics import (
    PolygonAveragePrecision,
    PolygonIOU,
    PolygonMAP,
    PolygonPrecision,
    PolygonRecall,
)
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
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
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
            PolygonPrecision,
            {"enforce_label_match": True},
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
    for label, result_val in result.items():
        assert_metric_eq(result_val, ScalarResult(1, 1))


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
            PolygonPrecision,
            {"enforce_label_match": True},
        ),
        (
            TEST_BOX_ANNOTATION_LIST,
            TEST_BOX_PREDICTION_LIST,
            PolygonRecall,
            {"enforce_label_match": True},
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
            PolygonPrecision,
            {"enforce_label_match": True},
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
    for label, result in result.items():
        assert_metric_eq(result, ScalarResult(0, 1))


@pytest.mark.parametrize(
    "test_annotations,test_predictions,metric_fn,expected,kwargs",
    [
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonIOU,
            {"car": ScalarResult(109.0 / 300, 3)},
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonPrecision,
            {"car": ScalarResult(1.0 / 3, 3)},
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonRecall,
            {"car": ScalarResult(0.5, 2)},
            {"enforce_label_match": True},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonAveragePrecision,
            {"car": ScalarResult(1.0 / 6, 1)},
            {"label": "car"},
        ),
        (
            TEST_ANNOTATION_LIST,
            TEST_PREDICTION_LIST,
            PolygonMAP,
            {"car": ScalarResult(1.0 / 6, 1)},
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
    for label, value in result.items():
        assert label in expected
        assert_metric_eq(value, expected[label])
