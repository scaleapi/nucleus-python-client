import numpy as np
import pytest

from nucleus.metrics.geometry import (
    GeometryPoint,
    GeometryPolygon,
    convex_polygon_intersection_area,
    polygon_intersection_area,
    segment_intersection,
)

RECTANGLE1 = GeometryPolygon(
    [
        GeometryPoint((0, 0)),
        GeometryPoint((100, 0)),
        GeometryPoint((100, 100)),
        GeometryPoint((0, 100)),
    ]
)

RECTANGLE2 = GeometryPolygon(
    [
        GeometryPoint((50, 50)),
        GeometryPoint((50, 150)),
        GeometryPoint((150, 150)),
        GeometryPoint((150, 50)),
    ]
)

SEGMENT1 = (GeometryPoint((0, 0)), GeometryPoint((0, 50)))

SEGMENT2 = (GeometryPoint((-25, 25)), GeometryPoint((25, 25)))


def test_segment_intersection():
    alpha, beta, intersection = segment_intersection(SEGMENT1, SEGMENT2)
    assert alpha == pytest.approx(0.5)
    assert beta == pytest.approx(0.5)
    assert intersection.x == pytest.approx(0)
    assert intersection.y == pytest.approx(25)


def test_convex_polygon_intersection_area():
    intersection = np.abs(
        convex_polygon_intersection_area(RECTANGLE1, RECTANGLE2)
    )
    assert intersection == pytest.approx(2500)


def test_polygon_intersection_area():
    intersection = polygon_intersection_area(RECTANGLE1, RECTANGLE2)
    assert intersection == pytest.approx(2500)


def test_polygon_area():
    assert RECTANGLE1.signed_area == -10000
    assert RECTANGLE1.area == 10000
    assert RECTANGLE2.signed_area == 10000
    assert RECTANGLE2.area == 10000
