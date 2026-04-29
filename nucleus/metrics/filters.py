from typing import List

from nucleus.prediction import PredictionList

from .custom_types import BoxOrPolygonAnnoOrPred
from .polygon_utils import polygon_annotation_to_shape


def polygon_area_filter(
    polygons: List[BoxOrPolygonAnnoOrPred], min_area: float, max_area: float
) -> List[BoxOrPolygonAnnoOrPred]:
    def _in_area_range(polygon: BoxOrPolygonAnnoOrPred) -> bool:
        return min_area <= polygon_annotation_to_shape(polygon) <= max_area

    return list(filter(_in_area_range, polygons))


def confidence_filter(
    predictions: PredictionList, min_confidence: float
) -> PredictionList:
    def _meets_min_confidence(prediction) -> bool:
        return not hasattr(prediction, "confidence") or (
            prediction.confidence >= min_confidence
        )

    predictions_copy = PredictionList()
    for attr in predictions.__dict__:
        predictions_copy.__dict__[attr] = list(
            filter(_meets_min_confidence, predictions.__dict__[attr])
        )
    return predictions_copy


def polygon_label_filter(
    polygons: List[BoxOrPolygonAnnoOrPred], label: str
) -> List[BoxOrPolygonAnnoOrPred]:
    return list(filter(lambda polygon: polygon.label == label, polygons))
