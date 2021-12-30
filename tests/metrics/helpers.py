import pytest

from nucleus.annotation import BoxAnnotation, PolygonAnnotation
from nucleus.metrics.base import Metric, MetricResult
from tests.helpers import TEST_BOX_ANNOTATIONS, TEST_CONVEX_POLYGON_ANNOTATIONS

TEST_BOX_ANNOTATIONS = [
    BoxAnnotation(**annotation) for annotation in TEST_BOX_ANNOTATIONS
]


TEST_CONVEX_POLYGON_ANNOTATIONS = [
    PolygonAnnotation.from_json(annotation)
    for annotation in TEST_CONVEX_POLYGON_ANNOTATIONS
]


def assert_metric_eq(actual, expected):
    assert expected.value == pytest.approx(actual.value)
    assert expected.weight == pytest.approx(actual.weight)


def perfect_match_test_wrapper(test_data, metric_fn):
    metric = metric_fn(enforce_label_match=False)
    result = metric(test_data, test_data)
    assert_metric_eq(result, MetricResult(1, len(test_data)))

    metric = metric_fn(enforce_label_match=True)
    result = metric(test_data, test_data)
    assert_metric_eq(result, MetricResult(1, len(test_data)))
