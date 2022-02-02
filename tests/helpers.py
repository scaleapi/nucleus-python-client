import math
import os
import uuid
from pathlib import Path

from nucleus import BoxPrediction, DatasetItem
from nucleus.modelci import ThresholdComparison

PRESIGN_EXPIRY_SECONDS = 60 * 60 * 24 * 2  # 2 days
N_UUID_CHARACTERS = 10
TEST_MODEL_NAME = "[PyTest] Test Model"
TEST_MODEL_RUN = "[PyTest] Test Model Run"
TEST_DATASET_NAME = "[PyTest] Test Dataset"
TEST_DATASET_3D_NAME = "[PyTest] Test Dataset 3D"
TEST_SLICE_NAME = "[PyTest] Test Slice"
TEST_PROJECT_ID = "60b699d70f139e002dd31bfc"

DATASET_WITH_AUTOTAG = "ds_c5jwptkgfsqg0cs503z0"
NUCLEUS_PYTEST_USER_ID = "60ad648c85db770026e9bf77"

EVAL_FUNCTION_THRESHOLD = 0.5
EVAL_FUNCTION_COMPARISON = ThresholdComparison.GREATER_THAN


TEST_IMG_URLS = [
    "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/airplane.jpeg",
    "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/arctichare.jpeg",
    "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/baboon.jpeg",
    "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/barbara.jpeg",
    "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/cat.jpeg",
]

TEST_POINTCLOUD_URLS = [
    "https://scaleapi-cust-lidar.s3.us-west-1.amazonaws.com/test-scale/frame-0.json",
    "https://scaleapi-cust-lidar.s3.us-west-1.amazonaws.com/test-scale/frame-1.json",
    "https://scaleapi-cust-lidar.s3.us-west-1.amazonaws.com/test-scale/frame-2.json",
    "https://scaleapi-cust-lidar.s3.us-west-1.amazonaws.com/test-scale/frame-3.json",
    "https://scaleapi-cust-lidar.s3.us-west-1.amazonaws.com/test-scale/frame-4.json",
]

TEST_LIDAR_SCENES = {
    "scenes": [
        {
            "reference_id": "scene_1",
            "frames": [
                {
                    "lidar": {
                        "pointcloud_url": TEST_POINTCLOUD_URLS[0],
                        "reference_id": "lidar_frame_1",
                        "metadata": {"frame": 1},
                    },
                    "camera": {
                        "image_url": TEST_IMG_URLS[0],
                        "reference_id": "camera_frame_1",
                        "metadata": {
                            "camera_params": {
                                "position": {"x": 0, "y": 0, "z": 0},
                                "heading": {
                                    "w": 0.6,
                                    "x": -0.6,
                                    "y": 0.3,
                                    "z": -0.3,
                                },
                                "fx": 1000,
                                "fy": 1000,
                                "cx": 1000,
                                "cy": 500,
                            }
                        },
                    },
                },
                {
                    "lidar": {
                        "pointcloud_url": TEST_POINTCLOUD_URLS[0],
                        "reference_id": "lidar_frame_2",
                        "metadata": {"frame": 2},
                    },
                },
            ],
            "metadata": {"meta_int": 123, "meta_str": "foo"},
        },
    ],
    "update": False,
}


def reference_id_from_url(url):
    return Path(url).name


TEST_DATASET_ITEMS = [
    DatasetItem(TEST_IMG_URLS[0], reference_id_from_url(TEST_IMG_URLS[0])),
    DatasetItem(TEST_IMG_URLS[1], reference_id_from_url(TEST_IMG_URLS[1])),
    DatasetItem(TEST_IMG_URLS[2], reference_id_from_url(TEST_IMG_URLS[2])),
    DatasetItem(TEST_IMG_URLS[3], reference_id_from_url(TEST_IMG_URLS[3])),
]

TEST_LIDAR_ITEMS = [
    DatasetItem(pointcloud_location=TEST_POINTCLOUD_URLS[0], reference_id="1"),
    DatasetItem(pointcloud_location=TEST_POINTCLOUD_URLS[1], reference_id="2"),
    DatasetItem(pointcloud_location=TEST_POINTCLOUD_URLS[2], reference_id="3"),
    DatasetItem(pointcloud_location=TEST_POINTCLOUD_URLS[3], reference_id="4"),
    DatasetItem(pointcloud_location=TEST_POINTCLOUD_URLS[4], reference_id="5"),
]

LOCAL_FILENAME = "tests/test_img.jpg"
TEST_PREDS = [
    BoxPrediction("[Pytest Box Prediction 1]", 0, 0, 100, 100, "1"),
    BoxPrediction("[Pytest Box Prediction 2]", 0, 0, 100, 100, "2"),
    BoxPrediction("[Pytest Box Prediction 3]", 0, 0, 100, 100, "3"),
    BoxPrediction("[Pytest Box Prediction 4]", 0, 0, 100, 100, "4"),
]


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
        "geometry": {
            "vertices": [
                {
                    "x": 50 + i * 10 + j,
                    "y": 60 + i * 10 + j,
                }
                for j in range(3)
            ],
        },
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Polygon Annotation Annotation Id{i}",
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_CONVEX_POLYGON_ANNOTATIONS = [
    {
        "label": f"[Pytest] Polygon Annotation ${i}",
        "geometry": {
            "vertices": [
                {
                    "x": 50 + i * 10 + math.cos(2 * j * math.pi / 3),
                    "y": 60 + i * 10 + math.sin(2 * j * math.pi / 3),
                }
                for j in range(3)
            ],
        },
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Polygon Annotation Annotation Id{i}",
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_CUBOID_ANNOTATIONS = [
    {
        "label": f"[Pytest] Cuboid Annotation #{i}",
        "geometry": {
            "position": {
                "x": 50 * i + 5,
                "y": 60 * i + 5,
                "z": 70 * i + 5,
            },
            "dimensions": {
                "x": 10 * i + 10,
                "y": 20 * i + 10,
                "z": 30 * i + 10,
            },
            "yaw": 5 * i,
        },
        "annotation_id": f"[Pytest] Cuboid Annotation Annotation Id {i}",
    }
    for i in range(len(TEST_POINTCLOUD_URLS))
]

TEST_CATEGORY_ANNOTATIONS = [
    {
        "label": f"[Pytest] Category Label ${i}",
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "taxonomy_name": "[Pytest] Category Taxonomy 1",
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_DEFAULT_CATEGORY_ANNOTATIONS = [
    {
        "label": f"[Pytest] Category Label ${i}",
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION = [
    {
        "label": "[Pytest] Category Label 0",
        "reference_id": reference_id_from_url(TEST_IMG_URLS[0]),
        "taxonomy_name": "[Pytest] Category Taxonomy Nonexistent",
    }
]

TEST_MULTICATEGORY_ANNOTATIONS = [
    {
        "labels": [
            f"[Pytest] MultiCategory Label ${i}",
            f"[Pytest] MultiCategory Label ${i+1}",
        ],
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "taxonomy_name": "[Pytest] MultiCategory Taxonomy 1",
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS = [
    {
        "labels": [
            f"[Pytest] MultiCategory Label ${i}",
            f"[Pytest] MultiCategory Label ${i+1}",
        ],
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
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

TEST_CATEGORY_MODEL_PDF = {
    category_annotation["label"]: 1 / len(TEST_CATEGORY_ANNOTATIONS)
    for category_annotation in TEST_CATEGORY_ANNOTATIONS
}

TEST_BOX_PREDICTIONS = [
    {
        **TEST_BOX_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_BOX_MODEL_PDF,
    }
    if i != 0
    else {
        **TEST_BOX_ANNOTATIONS[i],
        "confidence": 0.10 * i,
    }
    for i in range(len(TEST_BOX_ANNOTATIONS))
]

TEST_POLYGON_PREDICTIONS = [
    {
        **TEST_POLYGON_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_POLYGON_MODEL_PDF,
    }
    if i != 0
    else {
        **TEST_POLYGON_ANNOTATIONS[i],
        "confidence": 0.10 * i,
    }
    for i in range(len(TEST_POLYGON_ANNOTATIONS))
]

TEST_CATEGORY_PREDICTIONS = [
    {
        **TEST_CATEGORY_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_CATEGORY_MODEL_PDF,
    }
    if i != 0
    else {
        **TEST_CATEGORY_ANNOTATIONS[i],
        "confidence": 0.10 * i,
    }
    for i in range(len(TEST_CATEGORY_ANNOTATIONS))
]

TEST_DEFAULT_CATEGORY_PREDICTIONS = [
    {
        **TEST_DEFAULT_CATEGORY_ANNOTATIONS[i],
        "confidence": 0.10 * i,
        "class_pdf": TEST_CATEGORY_MODEL_PDF,
    }
    if i != 0
    else {
        **TEST_DEFAULT_CATEGORY_ANNOTATIONS[i],
        "confidence": 0.10 * i,
    }
    for i in range(len(TEST_DEFAULT_CATEGORY_ANNOTATIONS))
]

TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION = [
    {
        "label": "[Pytest] Category Label 0",
        "reference_id": reference_id_from_url(TEST_IMG_URLS[0]),
        "taxonomy_name": "[Pytest] Category Taxonomy Nonexistent",
        "confidence": 0.10,
    }
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
        annotation_instance.vertices, annotation_dict["geometry"]["vertices"]
    ):
        assert instance_pt.x == dict_pt["x"]
        assert instance_pt.y == dict_pt["y"]


def assert_cuboid_annotation_matches_dict(
    annotation_instance, annotation_dict
):
    assert annotation_instance.label == annotation_dict["label"]
    assert (
        annotation_instance.annotation_id == annotation_dict["annotation_id"]
    )

    instance_pos = annotation_instance.position
    dict_pos = annotation_dict["geometry"]["position"]
    assert instance_pos.x == dict_pos["x"]
    assert instance_pos.y == dict_pos["y"]
    assert instance_pos.z == dict_pos["z"]

    instance_dim = annotation_instance.dimensions
    dict_dim = annotation_dict["geometry"]["dimensions"]
    assert instance_dim.x == dict_dim["x"]
    assert instance_dim.y == dict_dim["y"]
    assert instance_dim.z == dict_dim["z"]

    assert annotation_instance.yaw == annotation_dict["geometry"]["yaw"]


def assert_category_annotation_matches_dict(
    annotation_instance, annotation_dict
):
    assert annotation_instance.label == annotation_dict["label"]
    if annotation_instance.taxonomy_name:
        assert annotation_instance.taxonomy_name == annotation_dict.get(
            "taxonomy_name", None
        )


def assert_multicategory_annotation_matches_dict(
    annotation_instance, annotation_dict
):
    assert set(annotation_instance.labels) == set(annotation_dict["labels"])
    if annotation_instance.taxonomy_name:
        assert annotation_instance.taxonomy_name == annotation_dict.get(
            "taxonomy_name", None
        )


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


def assert_category_prediction_matches_dict(
    prediction_instance, prediction_dict
):
    assert_category_annotation_matches_dict(
        prediction_instance, prediction_dict
    )
    assert prediction_instance.confidence == prediction_dict["confidence"]


def get_uuid():
    return str(uuid.uuid4())[-N_UUID_CHARACTERS:]


def running_as_nucleus_pytest_user(client):
    if NUCLEUS_PYTEST_USER_ID in client.api_key:
        return True
    if os.environ.get("NUCLEUS_PYTEST_USER_ID") == NUCLEUS_PYTEST_USER_ID:
        return True
    return False
