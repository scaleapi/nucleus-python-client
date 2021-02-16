import pytest

from helpers import (
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_BOX_ANNOTATIONS,
    TEST_POLYGON_ANNOTATIONS,
    reference_id_from_url,
    assert_box_annotation_matches_dict,
    assert_polygon_annotation_matches_dict,
)

from nucleus import BoxAnnotation, PolygonAnnotation, DatasetItem
from nucleus.constants import ERROR_PAYLOAD

@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(DatasetItem(
            image_location=url,
            reference_id=reference_id_from_url(url),
        ))

    response = ds.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {}


def test_box_gt_upload(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['dataset_id'] == dataset.id
    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(response_annotation, TEST_BOX_ANNOTATIONS[0])


def test_polygon_gt_upload(dataset):
    annotation = PolygonAnnotation(**TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['dataset_id'] == dataset.id
    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(response_annotation, TEST_POLYGON_ANNOTATIONS[0])


def test_box_gt_upload_update(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['annotations_processed'] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params['annotation_id'] = TEST_BOX_ANNOTATIONS[0]['annotation_id']
    annotation_update_params['reference_id'] = TEST_BOX_ANNOTATIONS[0]['reference_id']

    annotation_update = BoxAnnotation(**annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(response_annotation, annotation_update_params)


def test_box_gt_upload_ignore(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['annotations_processed'] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params['annotation_id'] = TEST_BOX_ANNOTATIONS[0]['annotation_id']
    annotation_update_params['reference_id'] = TEST_BOX_ANNOTATIONS[0]['reference_id']
    annotation_update = BoxAnnotation(**annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(response_annotation, TEST_BOX_ANNOTATIONS[0])


def test_polygon_gt_upload_update(dataset):
    annotation = PolygonAnnotation(**TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['annotations_processed'] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params['annotation_id'] = TEST_POLYGON_ANNOTATIONS[0]['annotation_id']
    annotation_update_params['reference_id'] = TEST_POLYGON_ANNOTATIONS[0]['reference_id']

    annotation_update = PolygonAnnotation(**annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(response_annotation, annotation_update_params)


def test_polygon_gt_upload_ignore(dataset):
    annotation = PolygonAnnotation(**TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response['annotations_processed'] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params['annotation_id'] = TEST_POLYGON_ANNOTATIONS[0]['annotation_id']
    annotation_update_params['reference_id'] = TEST_POLYGON_ANNOTATIONS[0]['reference_id']

    annotation_update = PolygonAnnotation(**annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response['annotations_processed'] == 1

    response = dataset.refloc(annotation.reference_id)['annotations']
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(response_annotation, TEST_POLYGON_ANNOTATIONS[0])