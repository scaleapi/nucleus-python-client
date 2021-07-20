import pytest

from .helpers import (
    TEST_DATASET_3D_NAME,
    TEST_CUBOID_ANNOTATIONS,
    TEST_LIDAR_FRAMES,
    assert_cuboid_annotation_matches_dict,
)

from nucleus import (
    CuboidAnnotation,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_scene_upload_sync(dataset):
    payload = TEST_LIDAR_FRAMES
    response = dataset.upload_scenes(payload)

    assert response["dataset_id"] == dataset.id
    assert response["new_items"] == 1
    assert response["updated_items"] == 0
    assert response["ignored_items"] == 0


@pytest.mark.integration
def test_scene_and_cuboid_upload_sync(dataset):
    payload = TEST_LIDAR_FRAMES
    response = dataset.upload_scenes(payload)

    assert response["dataset_id"] == dataset.id
    assert response["new_items"] == 1
    assert response["updated_items"] == 0
    assert response["ignored_items"] == 0

    TEST_CUBOID_ANNOTATIONS[0]["dataset_item_id"] = dataset.items[0].item_id
    annotation = CuboidAnnotation.from_json(TEST_CUBOID_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.loc(annotation.item_id)["annotations"]["cuboid"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_cuboid_annotation_matches_dict(
        response_annotation, TEST_CUBOID_ANNOTATIONS[0]
    )
