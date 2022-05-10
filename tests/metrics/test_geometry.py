import numpy as np
import pytest

try:
    from shapely.geometry import LineString, Polygon
except ModuleNotFoundError:
    pytest.skip(
        "Shapely not installed, skipping (install with poetry install -E shapely)",
        allow_module_level=True,
    )

RECTANGLE1 = Polygon(
    [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ],
)

RECTANGLE2 = Polygon(
    [
        (50, 50),
        (50, 150),
        (150, 150),
        (150, 50),
    ],
)

POLYGON1 = Polygon(
    [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ],
)

POLYGON2 = Polygon(
    [
        (50, 50),
        (50, 150),
        (150, 150),
        (150, 50),
    ],
)

SEGMENT1 = LineString([(0, 0), (0, 50)])

SEGMENT2 = LineString([(-25, 25), (25, 25)])


def test_segment_intersection():
    intersection = SEGMENT1.intersection(SEGMENT2)
    assert intersection.x == pytest.approx(0)
    assert intersection.y == pytest.approx(25)


def test_rectangle_intersection_area():
    intersection = RECTANGLE1.intersection(RECTANGLE2).area
    assert intersection == pytest.approx(2500)


def test_convex_polygon_intersection_area():
    intersection = np.abs(POLYGON1.intersection(POLYGON2).area)
    assert intersection == pytest.approx(2500)


def test_polygon_intersection_area():
    intersection = POLYGON1.intersection(POLYGON2).area
    assert intersection == pytest.approx(2500)


def test_polygon_area():
    assert RECTANGLE1.area == 10000
    assert RECTANGLE2.area == 10000
    assert POLYGON1.area == 10000
    assert POLYGON2.area == 10000
