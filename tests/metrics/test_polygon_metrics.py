from nucleus.metrics import PolygonIOU, PolygonPrecision, PolygonRecall
from tests.metrics.helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_CONVEX_POLYGON_ANNOTATIONS,
    perfect_match_test_wrapper,
)


def test_perfect_match_box_metrics():
    perfect_match_test_wrapper(TEST_BOX_ANNOTATIONS, PolygonIOU)
    perfect_match_test_wrapper(TEST_BOX_ANNOTATIONS, PolygonPrecision)
    perfect_match_test_wrapper(TEST_BOX_ANNOTATIONS, PolygonRecall)


def test_perfect_match_polygon_metrics():
    perfect_match_test_wrapper(TEST_CONVEX_POLYGON_ANNOTATIONS, PolygonIOU)
    perfect_match_test_wrapper(
        TEST_CONVEX_POLYGON_ANNOTATIONS, PolygonPrecision
    )
    perfect_match_test_wrapper(TEST_CONVEX_POLYGON_ANNOTATIONS, PolygonRecall)
