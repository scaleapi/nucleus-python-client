import pytest

from nucleus.annotation import CuboidAnnotation, Point3D
from nucleus.metrics.base import ScalarResult
from nucleus.metrics.cuboid_metrics import (
    CuboidIOU,
    CuboidPrecision,
    CuboidRecall,
)
from nucleus.prediction import CuboidPrediction

try:
    import shapely
except ModuleNotFoundError:
    pytest.skip(
        "Skipping metrics tests (to run install with poetry install -E shapely)",
        allow_module_level=True,
    )

CAR_LABEL = "car"
PEDESTRIAN_LABEL = "pedestrian"


def test_cuboid_metrics_simple():
    # single item, perfect predictions
    annotations = [
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(1000, 1000, 1000),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
    ]
    predictions = [
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(1000, 1000, 1000),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
    ]
    assert CuboidIOU().eval(annotations, predictions) == ScalarResult(
        1.0, len(annotations)
    ), "Unexpected Cuboid IoU result"
    assert CuboidPrecision().eval(annotations, predictions) == ScalarResult(
        1.0, len(annotations)
    ), "Unexpected Cuboid Precision result"
    assert CuboidRecall().eval(annotations, predictions) == ScalarResult(
        1.0, len(annotations)
    ), "Unexpected Cuboid Recall result"


def test_cuboid_metrics_numerical_check():
    # single item, realistic predictions w/ matches and non-matches
    annotations = [
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(1000, 1000, 1000),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(-100, -100, -100),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),  # false negative
    ]
    predictions = [
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(1.0, 1.0, 1.0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(999, 999, 999),
            dimensions=Point3D(8, 8, 6),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(250, 250, 250),
            dimensions=Point3D(2, 2, 2),
            yaw=0.0,
            reference_id="item_A",
        ),  # false positive
    ]
    cuboid_iou_result = CuboidIOU().eval(annotations, predictions)
    cuboid_precision_result = CuboidPrecision().eval(annotations, predictions)
    cuboid_recall_result = CuboidRecall().eval(annotations, predictions)
    assert cuboid_iou_result == ScalarResult(
        0.4316, 2
    ), f"Unexpected Cuboid IoU result: {cuboid_iou_result}"
    assert cuboid_precision_result == ScalarResult(
        2.0 / 3.0, len(predictions)
    ), f"Unexpected Cuboid Precision result {cuboid_precision_result}"
    assert cuboid_recall_result == ScalarResult(
        2.0 / 3.0, len(annotations)
    ), f"Unexpected Cuboid Recall result {cuboid_recall_result}"


def test_cuboid_metrics_class_labels():
    annotations = [
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(1000, 1000, 1000),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(-100, -100, -100),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),  # false negative
    ]
    predictions = [
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(1.0, 1.0, 1.0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=PEDESTRIAN_LABEL,
            position=Point3D(999, 999, 999),
            dimensions=Point3D(8, 8, 6),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(250, 250, 250),
            dimensions=Point3D(2, 2, 2),
            yaw=0.0,
            reference_id="item_A",
        ),  # false positive
    ]

    cuboid_iou_result1 = CuboidIOU().eval(annotations, predictions)
    cuboid_precision_result1 = CuboidPrecision().eval(annotations, predictions)
    cuboid_recall_result1 = CuboidRecall().eval(annotations, predictions)
    assert cuboid_iou_result1 == ScalarResult(
        0.47928, 1
    ), f"Unexpected Cuboid IoU result: {cuboid_iou_result1}"
    assert cuboid_precision_result1 == ScalarResult(
        1.0 / 3.0, len(predictions)
    ), f"Unexpected Cuboid Precision result {cuboid_precision_result1}"
    assert cuboid_recall_result1 == ScalarResult(
        1.0 / 3.0, len(annotations)
    ), f"Unexpected Cuboid Recall result {cuboid_recall_result1}"

    cuboid_iou_result2 = CuboidIOU(enforce_label_match=False).eval(
        annotations, predictions
    )
    cuboid_precision_result2 = CuboidPrecision(enforce_label_match=False).eval(
        annotations, predictions
    )
    cuboid_recall_result2 = CuboidRecall(enforce_label_match=False).eval(
        annotations, predictions
    )
    assert cuboid_iou_result2 == ScalarResult(
        0.4316, 2
    ), f"Unexpected Cuboid IoU result: {cuboid_iou_result2}"
    assert cuboid_precision_result2 == ScalarResult(
        2.0 / 3.0, len(predictions)
    ), f"Unexpected Cuboid Precision result {cuboid_precision_result2}"
    assert cuboid_recall_result2 == ScalarResult(
        2.0 / 3.0, len(annotations)
    ), f"Unexpected Cuboid Recall result {cuboid_recall_result2}"


def test_cuboid_metrics_multi_item():
    # single item, perfect precision
    annotations = [
        # first item
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(1000, 1000, 1000),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(-100, -100, -100),
            dimensions=Point3D(10, 10, 10),
            yaw=0.0,
            reference_id="item_A",
        ),  # false negative
        # second item
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(0, 0, 0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_B",
        ),
        CuboidAnnotation(
            label=CAR_LABEL,
            position=Point3D(30, 50, 120),
            dimensions=Point3D(1, 2.5, 3),
            yaw=0.0,
            reference_id="item_B",
        ),
    ]
    predictions = [
        # first item
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(1.0, 1.0, 1.0),
            dimensions=Point3D(10, 10, 5),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=PEDESTRIAN_LABEL,
            position=Point3D(999, 999, 999),
            dimensions=Point3D(8, 8, 6),
            yaw=0.0,
            reference_id="item_A",
        ),
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(250, 250, 250),
            dimensions=Point3D(2, 2, 2),
            yaw=0.0,
            reference_id="item_A",
        ),  # false positive
        # second item
        CuboidPrediction(
            label=CAR_LABEL,
            position=Point3D(250, 250, 250),
            dimensions=Point3D(2, 2, 2),
            yaw=0.0,
            reference_id="item_B",
        ),  # false positive
    ]

    cuboid_iou_result1 = CuboidIOU().eval(annotations, predictions)
    cuboid_precision_result1 = CuboidPrecision().eval(annotations, predictions)
    cuboid_recall_result1 = CuboidRecall().eval(annotations, predictions)
    assert cuboid_iou_result1 == ScalarResult(
        0.47928, 1
    ), f"Unexpected Cuboid IoU result: {cuboid_iou_result1}"
    assert cuboid_precision_result1 == ScalarResult(
        1.0 / len(predictions), len(predictions)
    ), f"Unexpected Cuboid Precision result {cuboid_precision_result1}"
    assert cuboid_recall_result1 == ScalarResult(
        1.0 / len(annotations), len(annotations)
    ), f"Unexpected Cuboid Recall result {cuboid_recall_result1}"

    cuboid_iou_result2 = CuboidIOU(enforce_label_match=False).eval(
        annotations, predictions
    )
    cuboid_precision_result2 = CuboidPrecision(enforce_label_match=False).eval(
        annotations, predictions
    )
    cuboid_recall_result2 = CuboidRecall(enforce_label_match=False).eval(
        annotations, predictions
    )
    assert cuboid_iou_result2 == ScalarResult(
        0.4316, 2
    ), f"Unexpected Cuboid IoU result: {cuboid_iou_result2}"
    assert cuboid_precision_result2 == ScalarResult(
        2.0 / len(predictions), len(predictions)
    ), f"Unexpected Cuboid Precision result {cuboid_precision_result2}"
    assert cuboid_recall_result2 == ScalarResult(
        2.0 / len(annotations), len(annotations)
    ), f"Unexpected Cuboid Recall result {cuboid_recall_result2}"
