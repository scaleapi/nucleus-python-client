from pathlib import Path
import time
from urllib.parse import urlparse

from nucleus import DatasetItem, BoxPrediction

PRESIGN_EXPIRY_SECONDS = 60 * 60 * 24 * 2  # 2 days

TEST_MODEL_NAME = "[PyTest] Test Model"
TEST_MODEL_RUN = "[PyTest] Test Model Run"
TEST_DATASET_NAME = "[PyTest] Test Dataset"
TEST_SLICE_NAME = "[PyTest] Test Slice"
TEST_PROJECT_ID = "60b699d70f139e002dd31bfc"


TEST_IMG_URLS = [
    "https://homepages.cae.wisc.edu/~ece533/images/airplane.png",
    "https://homepages.cae.wisc.edu/~ece533/images/arctichare.png",
    "https://homepages.cae.wisc.edu/~ece533/images/baboon.png",
    "https://homepages.cae.wisc.edu/~ece533/images/barbara.png",
    "https://homepages.cae.wisc.edu/~ece533/images/cat.png",
]

TEST_DATASET_ITEMS = [
    DatasetItem(TEST_IMG_URLS[0], "1"),
    DatasetItem(TEST_IMG_URLS[1], "2"),
    DatasetItem(TEST_IMG_URLS[2], "3"),
    DatasetItem(TEST_IMG_URLS[3], "4"),
]

LOCAL_FILENAME = "tests/test_img.jpg"
TEST_PREDS = [
    BoxPrediction("[Pytest Box Prediction 1]", 0, 0, 100, 100, "1"),
    BoxPrediction("[Pytest Box Prediction 2]", 0, 0, 100, 100, "2"),
    BoxPrediction("[Pytest Box Prediction 3]", 0, 0, 100, 100, "3"),
    BoxPrediction("[Pytest Box Prediction 4]", 0, 0, 100, 100, "4"),
]


def reference_id_from_url(url):
    return Path(url).name


TEST_BOX_ANNOTATIONS = [
    {
        "label": f"[Pytest] Box Annotation ${i}",
        "x": 50 + i * 10,
        "y": 60 + i * 10,
        "width": 70 + i * 10,
        "height": 80 + i * 10,
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Box Annotation Annotation Id{i}",
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_POLYGON_ANNOTATIONS = [
    {
        "label": f"[Pytest] Polygon Annotation ${i}",
        "vertices": [
            {
                "x": 50 + i * 10 + j,
                "y": 60 + i * 10 + j,
            }
            for j in range(3)
        ],
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Polygon Annotation Annotation Id{i}",
    }
    for i in range(len(TEST_IMG_URLS))
]


TEST_MASK_URL = "https://raw.githubusercontent.com/scaleapi/nucleus-python-client/master/tests/testdata/000000000285.png"

TEST_SEGMENTATION_ANNOTATIONS = [
    {
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Segmentation Annotation Id{i}",
        "mask_url": TEST_MASK_URL,
        "annotations": [
            {"label": "bear", "index": 2},
            {"label": "grass-merged", "index": 1},
        ],
    }
    for i in range(len(TEST_IMG_URLS))
]
TEST_SEGMENTATION_PREDICTIONS = TEST_SEGMENTATION_ANNOTATIONS

TEST_BOX_MODEL_PDF = {
    box_annotation["label"]: 1 / len(TEST_BOX_ANNOTATIONS)
    for box_annotation in TEST_BOX_ANNOTATIONS
}

TEST_POLYGON_MODEL_PDF = {
    polygon_annotation["label"]: 1 / len(TEST_POLYGON_ANNOTATIONS)
    for polygon_annotation in TEST_POLYGON_ANNOTATIONS
}

TEST_BOX_PREDICTIONS = [
    {
        **TEST_BOX_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_BOX_MODEL_PDF,
    }
    for i in range(len(TEST_BOX_ANNOTATIONS))
]

TEST_POLYGON_PREDICTIONS = [
    {
        **TEST_POLYGON_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_POLYGON_MODEL_PDF,
    }
    for i in range(len(TEST_POLYGON_ANNOTATIONS))
]

TEST_INDEX_EMBEDDINGS_FILE = "https://raw.githubusercontent.com/scaleapi/nucleus-python-client/master/tests/testdata/pytest_embeddings_payload.json"


# Asserts that a box annotation instance matches a dict representing its properties.
# Useful to check annotation uploads/updates match.
def assert_box_annotation_matches_dict(annotation_instance, annotation_dict):
    assert annotation_instance.label == annotation_dict["label"]
    assert annotation_instance.x == annotation_dict["x"]
    assert annotation_instance.y == annotation_dict["y"]
    assert annotation_instance.height == annotation_dict["height"]
    assert annotation_instance.width == annotation_dict["width"]
    assert (
        annotation_instance.annotation_id == annotation_dict["annotation_id"]
    )


def assert_polygon_annotation_matches_dict(
    annotation_instance, annotation_dict
):
    assert annotation_instance.label == annotation_dict["label"]
    assert (
        annotation_instance.annotation_id == annotation_dict["annotation_id"]
    )
    for instance_pt, dict_pt in zip(
        annotation_instance.vertices, annotation_dict["vertices"]
    ):
        assert instance_pt["x"] == dict_pt["x"]
        assert instance_pt["y"] == dict_pt["y"]


def assert_segmentation_annotation_matches_dict(
    annotation_instance, annotation_dict
):
    assert annotation_instance.mask_url == annotation_dict["mask_url"]
    assert (
        annotation_instance.annotation_id == annotation_dict["annotation_id"]
    )
    # Cannot guarantee segments are in same order
    assert len(annotation_instance.annotations) == len(
        annotation_dict["annotations"]
    )

    for instance_segment, dict_segment in zip(
        sorted(annotation_instance.annotations, key=lambda i: i.index),
        sorted(annotation_dict["annotations"], key=lambda i: i["index"]),
    ):
        assert instance_segment.index == dict_segment["index"]
        assert instance_segment.label == dict_segment["label"]


# Asserts that a box prediction instance matches a dict representing its properties.
# Useful to check prediction uploads/updates match.
def assert_box_prediction_matches_dict(prediction_instance, prediction_dict):
    assert_box_annotation_matches_dict(prediction_instance, prediction_dict)
    assert prediction_instance.confidence == prediction_dict["confidence"]


def assert_polygon_prediction_matches_dict(
    prediction_instance, prediction_dict
):
    assert_polygon_annotation_matches_dict(
        prediction_instance, prediction_dict
    )
    assert prediction_instance.confidence == prediction_dict["confidence"]
