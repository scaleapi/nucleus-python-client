from typing import List, Tuple

import numpy as np
from rasterio import features
from shapely.geometry import Polygon
from nucleus.annotation import PolygonAnnotation, Point, Segment
from nucleus.metrics.custom_types import BoxOrPolygonAnnotation
from nucleus.metrics.polygon_utils import polygon_annotation_to_shape


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
    rasterized = rasterize_polygons_to_segmentation_mask(annotations, shape)
    assert (
        rasterized[:50, :50] == 1
    ).all(), "Expected all values in top right corner to be == '1'"
    assert (
        rasterized[:50, 50:] == 2
    ).all(), "Expected all values in top left corner to be == '2'"


def rasterize_polygons_to_segmentation_mask(
    annotations: List[BoxOrPolygonAnnotation], shape: Tuple[int, int]
) -> np.ndarray:
    polys = [polygon_annotation_to_shape(a) for a in annotations]
    segments = [
        Segment(ann.label, index=idx + 1, metadata=ann.metadata)
        for idx, ann in enumerate(annotations)
    ]
    poly_vals = [
        (poly, segment.index) for poly, segment in zip(polys, segments)
    ]
    rasterized = features.rasterize(
        poly_vals,
        out_shape=shape,
        fill=0,
        out=None,
        all_touched=False,
        dtype=None,
    )
    return rasterized
