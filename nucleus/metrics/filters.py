from typing import List

from nucleus.prediction import PredictionList

from .polygon_utils import (
    BoxOrPolygonAnnoOrPred,
    polygon_annotation_to_geometry,
)


def polygon_area_filter(
    polygons: List[BoxOrPolygonAnnoOrPred], min_area: float, max_area: float
) -> List[BoxOrPolygonAnnoOrPred]:
    filter_fn = (
        lambda polygon: min_area
        <= polygon_annotation_to_geometry(polygon).signed_area
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


def polygon_label_filter(
    polygons: List[BoxOrPolygonAnnoOrPred], label: str
) -> List[BoxOrPolygonAnnoOrPred]:
    return list(filter(lambda polygon: polygon.label == label, polygons))
