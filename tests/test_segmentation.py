from nucleus.annotation import SegmentationAnnotation
from nucleus.dataset import Dataset
from nucleus.model import Model
from nucleus.prediction import SegmentationPrediction
from tests.helpers import (
    NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET,
    TEST_LOCAL_MASK_URL,
    TEST_SEGMENTATION_ANNOTATIONS,
    TEST_SEGMENTATION_PREDICTIONS,
    assert_segmentation_annotation_matches_dict,
)


def test_segmentation_pred_upload_local(dataset: Dataset, model: Model):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    prediction.mask_url = TEST_LOCAL_MASK_URL
    response = dataset.upload_predictions(model, [prediction])

    assert response["predictions_processed"] == 1

    response = dataset.predictions_refloc(model, prediction.reference_id)[
        "segmentation"
    ][0]
    assert isinstance(response, SegmentationPrediction)
    assert response == prediction


def test_segmentation_pred_upload(dataset: Dataset, model: Model):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    response = dataset.upload_predictions(model, [prediction])

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = dataset.predictions_refloc(model, prediction.reference_id)[
        "segmentation"
    ]
    assert isinstance(response[0], SegmentationPrediction)

    assert_segmentation_annotation_matches_dict(
        response[0], TEST_SEGMENTATION_PREDICTIONS[0]
    )


def test_segmentation_pred_upload_ignore(dataset: Dataset, model: Model):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    dataset.upload_predictions(model, [prediction])

    # Upload Duplicate annotation
    response = dataset.upload_predictions(model, [prediction])
    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1


def test_single_local_semseg_gt_upload(dataset: Dataset):
    request_annotation = SegmentationAnnotation.from_json(
        TEST_SEGMENTATION_ANNOTATIONS[0]
    )
    request_annotation.mask_url = TEST_LOCAL_MASK_URL
    response = dataset.annotate(annotations=[request_annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response_annotation = dataset.refloc(request_annotation.reference_id)[
        "annotations"
    ]["segmentation"][0]

    assert response_annotation == request_annotation


def test_batch_local_semseg_gt_upload(dataset: Dataset):

    # This reference id is not in the dataset.
    bad_reference_id = TEST_SEGMENTATION_ANNOTATIONS[-1]["reference_id"]

    request_annotations = [
        SegmentationAnnotation.from_json(json_data)
        for json_data in TEST_SEGMENTATION_ANNOTATIONS
    ]
    for request_annotation in request_annotations:
        request_annotation.mask_url = TEST_LOCAL_MASK_URL
    response = dataset.annotate(annotations=request_annotations)

    assert response["dataset_id"] == dataset.id
    assert (
        response["annotations_processed"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )
    assert response["annotations_ignored"] == 0
    assert bad_reference_id in response["errors"][0]

    for request_annotation in request_annotations[
        :NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    ]:
        response_annotation = dataset.refloc(request_annotation.reference_id)[
            "annotations"
        ]["segmentation"][0]

        assert response_annotation == request_annotation


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
    assert (
        response["annotations_processed"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )
    assert response["annotations_ignored"] == 0


def test_batch_semseg_gt_upload_ignore(dataset):
    # First upload annotations
    annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert (
        response["annotations_processed"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )
    assert response["annotations_ignored"] == 0

    # When we re-upload, expect them to be ignored
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 0
    assert (
        response["annotations_ignored"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )


def test_batch_semseg_gt_upload_update(dataset):
    # First upload annotations
    annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert (
        response["annotations_processed"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )
    assert response["annotations_ignored"] == 0

    # When we re-upload, expect uploads to be processed
    response = dataset.annotate(annotations=annotations, update=True)
    assert response["dataset_id"] == dataset.id
    assert (
        response["annotations_processed"]
        == NUM_VALID_SEGMENTATIONS_IN_MAIN_DATASET
    )
    assert response["annotations_ignored"] == 0
