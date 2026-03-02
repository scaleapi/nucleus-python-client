from typing import TypeVar

from nucleus.annotation import BoxAnnotation, PolygonAnnotation
from nucleus.prediction import BoxPrediction, PolygonPrediction

BoxOrPolygonPrediction = TypeVar(
    "BoxOrPolygonPrediction", BoxPrediction, PolygonPrediction
)
BoxOrPolygonAnnotation = TypeVar(
    "BoxOrPolygonAnnotation", BoxAnnotation, PolygonAnnotation
)
BoxOrPolygonAnnoOrPred = TypeVar(
    "BoxOrPolygonAnnoOrPred",
    BoxAnnotation,
    PolygonAnnotation,
    BoxPrediction,
    PolygonPrediction,
)
