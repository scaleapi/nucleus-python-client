from typing import List

from nucleus.prediction import PredictionList

from .polygon_utils import BoxOrPolygonAnnotation, polygon_annotation_to_shape


def polygon_area_filter(
    polygons: List[BoxOrPolygonAnnotation], min_area: float, max_area: float
) -> List[BoxOrPolygonAnnotation]:
    filter_fn = (
        lambda polygon: min_area
        <= polygon_annotation_to_shape(polygon)
        <= max_area
    )
    return list(filter(filter_fn, polygons))


def confidence_filter(
    predictions: PredictionList, min_confidence: float
) -> PredictionList:
    predictions_copy = PredictionList()
    filter_fn = (
        lambda prediction: not hasattr(prediction, "confidence")
        or prediction.confidence >= min_confidence
    )
    predictions_copy.box_predictions = list(
        filter(filter_fn, predictions.box_predictions)
    )
    predictions_copy.polygon_predictions = list(
        filter(filter_fn, predictions.polygon_predictions)
    )
    predictions_copy.cuboid_predictions = list(
        filter(filter_fn, predictions.cuboid_predictions)
    )
    predictions_copy.category_predictions = list(
        filter(filter_fn, predictions.category_predictions)
    )
    predictions_copy.segmentation_predictions = list(
        filter(filter_fn, predictions.segmentation_predictions)
    )
    return predictions_copy
