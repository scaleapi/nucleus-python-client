from nucleus.annotation import Point, PolygonAnnotation
from nucleus.metrics.segmentation_utils import (
    rasterize_polygons_to_segmentation_mask,
)


def test_rasterize_polygons():
    shape = (100, 100)
    annotations = [
        PolygonAnnotation(
            "item",
            [
                Point(0, 0),
                Point(50, 0),
                Point(50, 50),
                Point(0, 50),
                Point(0, 0),
            ],
            reference_id="test",
        ),
        PolygonAnnotation(
            "item",
            [
                Point(50, 0),
                Point(100, 0),
                Point(100, 50),
                Point(50, 50),
                Point(50, 0),
            ],
            reference_id="test",
        ),
    ]
    rasterized, segments = rasterize_polygons_to_segmentation_mask(
        annotations, shape
    )
    assert (
        rasterized[:50, :50] == segments[0].index
    ).all(), f"Expected all values in top right corner to be == '{segments[0].index}'"
    assert (
        rasterized[:50, 50:] == segments[1].index
    ).all(), f"Expected all values in top left corner to be == '{segments[1].index}'"
