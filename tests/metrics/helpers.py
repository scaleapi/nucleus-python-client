import pytest

from nucleus.annotation import AnnotationList, BoxAnnotation, PolygonAnnotation
from nucleus.prediction import BoxPrediction, PolygonPrediction, PredictionList
from tests.helpers import TEST_BOX_ANNOTATIONS, TEST_CONVEX_POLYGON_ANNOTATIONS

TEST_BOX_ANNOTATION_LIST = AnnotationList(
    box_annotations=[
        BoxAnnotation(**annotation) for annotation in TEST_BOX_ANNOTATIONS
    ]
)

TEST_BOX_PREDICTION_LIST = PredictionList(
    box_predictions=[
        BoxPrediction(confidence=0.5, **annotation)
        for annotation in TEST_BOX_ANNOTATIONS
    ]
)


TEST_CONVEX_POLYGON_ANNOTATION_LIST = AnnotationList(
    polygon_annotations=[
        PolygonAnnotation.from_json(annotation)
        for annotation in TEST_CONVEX_POLYGON_ANNOTATIONS
    ]
)

TEST_CONVEX_POLYGON_PREDICTION_LIST = PredictionList(
    polygon_predictions=[
        PolygonPrediction.from_json(dict(**annotation, confidence=0.5))
        for annotation in TEST_CONVEX_POLYGON_ANNOTATIONS
    ]
)


TEST_ANNOTATION_LIST = AnnotationList(
    box_annotations=[
        BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
        ),
        BoxAnnotation(
            label="car",
            x=5,
            y=5,
            width=3,
            height=3,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
        ),
    ]
)

TEST_PREDICTION_LIST = PredictionList(
    box_predictions=[
        BoxPrediction(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            confidence=0.6,
            annotation_id="image_1_car_box_1",
        ),
        BoxPrediction(
            label="car",
            x=5,
            y=5,
            width=10,
            height=10,
            reference_id="image_1",
            confidence=0.9,
            annotation_id="image_1_car_box_1",
        ),
        BoxPrediction(
            label="car",
            x=5,
            y=5,
            width=10,
            height=10,
            reference_id="image_1",
            confidence=0.9,
            annotation_id="image_1_car_box_1",
        ),
    ]
)


def assert_metric_eq(actual, expected):
    assert expected.value == pytest.approx(actual.value)
    assert expected.weight == pytest.approx(actual.weight)
