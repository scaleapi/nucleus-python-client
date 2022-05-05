import numpy as np
from rasterio import features

from nucleus import Point, PolygonPrediction

try:
    from shapely import geometry
except ModuleNotFoundError:
    from nucleus.shapely_not_installed import (  # pylint: disable=ungrouped-imports
        ShapelyNotInstalled,
    )

    geometry = ShapelyNotInstalled


def instance_mask_to_polys(instance_mask: np.ndarray, background_code=None):
    mask_values = []
    all_polygons = []
    not_background_mask = (
        (instance_mask != background_code) if background_code else None
    )
    for shape, value in features.shapes(
        instance_mask.astype(np.int16),
        mask=not_background_mask,
    ):
        poly = geometry.shape(shape)
        all_polygons.append(poly)
        mask_values.append(int(value))

    return mask_values, all_polygons


def transform_poly_codes_to_poly_preds(
    dataset_item_id: str, pred_value, pred_polys, code_to_label
):
    polygon_predictions = []
    for code, poly in zip(pred_value, pred_polys):
        if poly.type != "Polygon":
            continue
        label = code_to_label[code]
        x_stack, y_stack = poly.exterior.coords.xy
        pred = PolygonPrediction(
            label,
            vertices=[Point(x, y) for x, y in zip(x_stack, y_stack)],
            reference_id=dataset_item_id,
        )
        polygon_predictions.append(pred)
    return polygon_predictions
