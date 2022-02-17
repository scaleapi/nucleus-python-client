import numpy as np
import pytest

from nucleus.metrics.geometry import (
    GeometryPolygon,
    convex_polygon_intersection_area,
    polygon_intersection_area,
    segment_intersection,
)

RECTANGLE1 = GeometryPolygon(
    points=[
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ],
    is_rectangle=True,
)

RECTANGLE2 = GeometryPolygon(
    points=[
        (50, 50),
        (50, 150),
        (150, 150),
        (150, 50),
    ],
    is_rectangle=True,
)

POLYGON1 = GeometryPolygon(
    points=[
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ],
    is_rectangle=False,
)

POLYGON2 = GeometryPolygon(
    points=[
        (50, 50),
        (50, 150),
        (150, 150),
        (150, 50),
    ],
    is_rectangle=False,
)

SEGMENT1 = (np.array((0, 0)), np.array((0, 50)))

SEGMENT2 = (np.array((-25, 25)), np.array((25, 25)))


def test_segment_intersection():
    alpha, beta, intersection = segment_intersection(SEGMENT1, SEGMENT2)
    assert alpha == pytest.approx(0.5)
    assert beta == pytest.approx(0.5)
    assert intersection[0] == pytest.approx(0)
    assert intersection[1] == pytest.approx(25)


def test_rectangle_intersection_area():
    intersection = polygon_intersection_area(RECTANGLE1, RECTANGLE2)
    assert intersection == pytest.approx(2500)


def test_convex_polygon_intersection_area():
    intersection = np.abs(convex_polygon_intersection_area(POLYGON1, POLYGON2))
    assert intersection == pytest.approx(2500)


def test_polygon_intersection_area():
    intersection = polygon_intersection_area(POLYGON1, POLYGON2)
    assert intersection == pytest.approx(2500)


def test_polygon_area():
    assert RECTANGLE1.signed_area == 10000
    assert RECTANGLE1.area == 10000
    assert RECTANGLE2.signed_area == 10000
    assert RECTANGLE2.area == 10000
    assert POLYGON1.signed_area == 10000
    assert POLYGON1.area == 10000
    assert POLYGON2.signed_area == -10000
    assert POLYGON2.area == 10000
