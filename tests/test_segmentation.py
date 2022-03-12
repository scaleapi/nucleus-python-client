from nucleus.annotation import SegmentationAnnotation
from tests.helpers import (
    TEST_LOCAL_MASK_URL,
    TEST_SEGMENTATION_ANNOTATIONS,
    assert_segmentation_annotation_matches_dict,
)


def test_single_local_semseg_gt_upload(dataset):
    annotation = SegmentationAnnotation.from_json(
        TEST_SEGMENTATION_ANNOTATIONS[0]
    )
    annotation.mask_url = TEST_LOCAL_MASK_URL
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
