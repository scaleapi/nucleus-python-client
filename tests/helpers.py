from pathlib import Path
from urllib.parse import urlparse
import boto3
from nucleus import DatasetItem, BoxPrediction

PRESIGN_EXPIRY_SECONDS = 60 * 60 * 24 * 2  # 2 days

TEST_MODEL_NAME = "[PyTest] Test Model"
TEST_MODEL_REFERENCE = "[PyTest] Test Model Reference"
TEST_MODEL_RUN = "[PyTest] Test Model Run"
TEST_DATASET_NAME = "[PyTest] Test Dataset"
TEST_SLICE_NAME = "[PyTest] Test Slice"

TEST_MODEL_NAME = "[PyTest] Test Model Name"
TEST_MODEL_REFERENCE = "[PyTest] Test Model Reference"
TEST_MODEL_RUN = "[PyTest] Test Model Run Reference"
TEST_DATASET_NAME = "[PyTest] Test Dataset"
TEST_SLICE_NAME = "[PyTest] Test Slice"
TEST_IMG_URLS = [
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/6dd63871-831611a6.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/82c1005c-e2d1d94f.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/7f2e1814-6591087d.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/06924f46-1708b96f.jpg",
    "s3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/89b42832-10d662f4.jpg",
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


def get_signed_url(url):
    bucket, key = get_s3_details(url)
    return s3_sign(bucket, key)


def get_s3_details(url):
    # Expects S3 URL format to be https://<BUCKET>.s3.amazonaws.com/<KEY>
    parsed = urlparse(url)
    bucket = parsed.netloc[: parsed.netloc.find(".")]
    return bucket, parsed.path[1:]


def s3_sign(bucket, key):
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
        },
        ExpiresIn=PRESIGN_EXPIRY_SECONDS,
    )


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

TEST_MASK_URL = "https://scale-ml.s3.amazonaws.com/home/nucleus/mscoco_masks_uint8/000000000285.png"
TEST_SEGMENTATION_ANNOTATIONS = [
    {
        "reference_id": reference_id_from_url(TEST_IMG_URLS[i]),
        "annotation_id": f"[Pytest] Segmentation Annotation Id{i}",
        "mask_url": get_signed_url(TEST_MASK_URL),
        "annotations": [
            {"label": "bear", "index": 2},
            {"label": "grass-merged", "index": 1},
        ],
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_SEGMENTATION_PREDICTIONS = TEST_SEGMENTATION_ANNOTATIONS

TEST_BOX_PREDICTIONS = [
    {**TEST_BOX_ANNOTATIONS[i], "confidence": 0.10 * i}
    for i in range(len(TEST_BOX_ANNOTATIONS))
]

TEST_POLYGON_PREDICTIONS = [
    {**TEST_POLYGON_ANNOTATIONS[i], "confidence": 0.10 * i}
    for i in range(len(TEST_POLYGON_ANNOTATIONS))
]

TEST_INDEX_EMBEDDINGS_FILE = "https://scale-ml.s3.amazonaws.com/home/nucleus/pytest/pytest_embeddings_payload.json"

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