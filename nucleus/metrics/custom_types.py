from typing import TypeVar

from nucleus import (
    BoxAnnotation,
    BoxPrediction,
    PolygonAnnotation,
    PolygonPrediction,
)

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
