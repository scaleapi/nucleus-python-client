import pytest
from nucleus.constants import (
    ANNOTATIONS_KEY,
    DATASET_ITEM_ID_KEY,
    FRAMES_KEY,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    ITEM_KEY,
    POINTCLOUD_KEY,
    POINTCLOUD_URL_KEY,
    REFERENCE_ID_KEY,
    SCENES_KEY,
    UPDATE_KEY,
)

from nucleus import (
    CuboidAnnotation,
    LidarScene,
    Frame,
)

from .helpers import (
    TEST_DATASET_3D_NAME,
    TEST_CUBOID_ANNOTATIONS,
    TEST_DATASET_ITEMS,
    TEST_LIDAR_ITEMS,
    TEST_LIDAR_SCENES,
    assert_cuboid_annotation_matches_dict,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_frame_add_item(dataset):
    frame = Frame(index=0)
    frame.add_item(TEST_DATASET_ITEMS[0], "camera")
    frame.add_item(TEST_LIDAR_ITEMS[0], "lidar")

    assert frame.get_index() == 0
    assert frame.get_sensors() == ["camera", "lidar"]
    for item in frame.get_items():
        assert item in [TEST_DATASET_ITEMS[0], TEST_LIDAR_ITEMS[0]]
    assert frame.get_item("lidar") == TEST_LIDAR_ITEMS[0]
    assert frame.to_payload() == {
        "camera": {
            "url": TEST_DATASET_ITEMS[0].image_location,
            "reference_id": TEST_DATASET_ITEMS[0].reference_id,
            "type": IMAGE_KEY,
            "metadata": TEST_DATASET_ITEMS[0].metadata or {},
        },
        "lidar": {
            "url": TEST_LIDAR_ITEMS[0].pointcloud_location,
            "reference_id": TEST_LIDAR_ITEMS[0].reference_id,
            "type": POINTCLOUD_KEY,
            "metadata": TEST_LIDAR_ITEMS[0].metadata or {},
        },
    }


def test_scene_upload_sync(dataset):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    response = dataset.append(scenes, update=update)

    assert response["dataset_id"] == dataset.id
    assert response["new_scenes"] == len(scenes)


@pytest.mark.integration
def test_scene_and_cuboid_upload_sync(dataset):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    response = dataset.append(scenes, update=update)

    assert response["dataset_id"] == dataset.id
    assert response["new_scenes"] == len(scenes)

    lidar_item_ref = payload[SCENES_KEY][0][FRAMES_KEY][0]["lidar"][
        REFERENCE_ID_KEY
    ]
    TEST_CUBOID_ANNOTATIONS[0][REFERENCE_ID_KEY] = lidar_item_ref

    annotations = [CuboidAnnotation.from_json(TEST_CUBOID_ANNOTATIONS[0])]
    response = dataset.annotate(annotations)

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == len(annotations)
    assert response["annotations_ignored"] == 0

    response_annotations = dataset.refloc(lidar_item_ref)[ANNOTATIONS_KEY][
        "cuboid"
    ]
    assert len(response_annotations) == 1
    assert_cuboid_annotation_matches_dict(
        response_annotations[0], TEST_CUBOID_ANNOTATIONS[0]
    )


@pytest.mark.integration
def test_scene_upload_async(dataset):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    assert status == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "SceneUploadResponse": {
                "errors": [],
                "dataset_id": dataset.id,
                "new_scenes": len(scenes),
                "ignored_scenes": 0,
                "scenes_errored": 0,
            }
        },
    }
