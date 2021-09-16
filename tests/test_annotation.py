import pytest

from .helpers import (
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_BOX_ANNOTATIONS,
    TEST_POLYGON_ANNOTATIONS,
    TEST_CATEGORY_ANNOTATIONS,
    TEST_SEGMENTATION_ANNOTATIONS,
    reference_id_from_url,
    assert_box_annotation_matches_dict,
    assert_polygon_annotation_matches_dict,
    assert_category_annotation_matches_dict,
    assert_segmentation_annotation_matches_dict,
)

from nucleus import (
    BoxAnnotation,
    PolygonAnnotation,
    CategoryAnnotation,
    SegmentationAnnotation,
    DatasetItem,
    Segment,
    Point,
)
from nucleus.constants import ERROR_PAYLOAD


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    [
        test_repr(SegmentationAnnotation.from_json(_))
        for _ in TEST_SEGMENTATION_ANNOTATIONS
    ]

    [test_repr(BoxAnnotation.from_json(_)) for _ in TEST_BOX_ANNOTATIONS]

    [
        test_repr(PolygonAnnotation.from_json(_))
        for _ in TEST_POLYGON_ANNOTATIONS
    ]


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )

    response = ds.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_box_gt_upload(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    print(annotation)
    response = dataset.annotate(annotations=[annotation])
    print(response)

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, TEST_BOX_ANNOTATIONS[0]
    )


def test_polygon_gt_upload(dataset):
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "polygon"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, TEST_POLYGON_ANNOTATIONS[0]
    )


def test_category_gt_upload(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, TEST_CATEGORY_ANNOTATIONS[0]
    )


def test_single_semseg_gt_upload(dataset):
    annotation = SegmentationAnnotation.from_json(
        TEST_SEGMENTATION_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response_annotation = dataset.refloc(annotation.reference_id)[
        "annotations"
    ]["segmentation"][0]
    assert_segmentation_annotation_matches_dict(
        response_annotation, TEST_SEGMENTATION_ANNOTATIONS[0]
    )


def test_batch_semseg_gt_upload(dataset):
    annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 5
    assert response["annotations_ignored"] == 0


def test_batch_semseg_gt_upload_ignore(dataset):
    # First upload annotations
    annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 5
    assert response["annotations_ignored"] == 0

    # When we re-upload, expect them to be ignored
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 5


def test_batch_semseg_gt_upload_update(dataset):
    # First upload annotations
    annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 5
    assert response["annotations_ignored"] == 0

    # When we re-upload, expect uploads to be processed
    response = dataset.annotate(annotations=annotations, update=True)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 5
    assert response["annotations_ignored"] == 0


def test_mixed_annotation_upload(dataset):
    # First upload annotations
    semseg_annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    bbox_annotations = [BoxAnnotation(**(ann)) for ann in TEST_BOX_ANNOTATIONS]
    annotations = bbox_annotations + semseg_annotations
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 10
    assert response["annotations_ignored"] == 0
    response_annotations = dataset.refloc(bbox_annotations[0].reference_id)[
        "annotations"
    ]
    assert len(response_annotations) == 2
    assert len(response_annotations["box"]) == 1
    assert "segmentation" in response_annotations


def test_box_gt_upload_update(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_BOX_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_BOX_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = BoxAnnotation(**annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_box_gt_upload_ignore(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])

    print(annotation)

    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_BOX_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_BOX_ANNOTATIONS[0][
        "reference_id"
    ]
    annotation_update = BoxAnnotation(**annotation_update_params)

    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, TEST_BOX_ANNOTATIONS[0]
    )


def test_polygon_gt_upload_update(dataset):
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = PolygonAnnotation.from_json(annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "polygon"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_polygon_gt_upload_ignore(dataset):
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = PolygonAnnotation.from_json(annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "polygon"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, TEST_POLYGON_ANNOTATIONS[0]
    )


def test_category_gt_upload_update(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_CATEGORY_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_category_gt_upload_ignore(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_CATEGORY_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, TEST_CATEGORY_ANNOTATIONS[0]
    )

    @pytest.mark.integration
    def test_box_gt_deletion(dataset):
        annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])

        print(annotation)

        response = dataset.annotate(annotations=[annotation])

        assert response["annotations_processed"] == 1

        job = dataset.delete_annotations()
        job.sleep_until_complete()
        job_status = job.status()
        assert job_status["status"] == "Completed"
        assert job_status["job_id"] == job.id
