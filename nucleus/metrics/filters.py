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
    for attr in predictions.__dict__:
        predictions_copy.__dict__[attr] = list(
            filter(filter_fn, predictions.__dict__[attr])
        )
    return predictions_copy
