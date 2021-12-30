from typing import List, Tuple

from nucleus.annotation import Annotation
from nucleus.prediction import Prediction

from .base import AnnotationOrPredictionList, Filter, PredictionFilter
from .polygon_utils import BoxOrPolygonAnnotation, _polygon_annotation_to_shape


class TypeFilter(Filter):
    """Filters annotations and predictions by type (e.g. Box, Polygon, Cuboid)."""

    def __init__(self, types: Tuple[type, ...]):
        assert all(issubclass(typ, Annotation) for typ in types)
        self.filter_fn = lambda x: isinstance(x, types)

    def __call__(
        self, annotations: AnnotationOrPredictionList
    ) -> AnnotationOrPredictionList:
        return list(filter(self.filter_fn, annotations))


class PolygonAreaFilter(Filter):
    """Filters polygon annotations and predictions by area."""

    def __init__(self, min_area: float, max_area: float):
        self.min_area = min_area
        self.max_area = max_area

    def filter_fn(self, annotation: BoxOrPolygonAnnotation):
        area = _polygon_annotation_to_shape(annotation).area
        return self.min_area <= area <= self.max_area

    def __call__(self, annotations: List[BoxOrPolygonAnnotation]) -> List[BoxOrPolygonAnnotation]:  # type: ignore[override]
        return list(filter(self.filter_fn, annotations))


class ConfidenceFilter(PredictionFilter):
    """Filters predictions by confidence score."""

    def __init__(self, min_confidence: float):
        assert (
            0 <= min_confidence <= 1
        ), "Min confidence must be between 0 and 1."
        self.filter_fn = lambda x: x.confidence >= min_confidence

    def __call__(self, predictions: List[Prediction]) -> List[Prediction]:
        return list(filter(self.filter_fn, predictions))
