import copy
import time

import pytest

from nucleus import CuboidAnnotation, DatasetItem, Frame, LidarScene
from nucleus.constants import (
    ANNOTATIONS_KEY,
    FRAMES_KEY,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    LENGTH_KEY,
    METADATA_KEY,
    NUM_SENSORS_KEY,
    POINTCLOUD_KEY,
    POINTCLOUD_URL_KEY,
    REFERENCE_ID_KEY,
    SCENES_KEY,
    TYPE_KEY,
    UPDATE_KEY,
    URL_KEY,
)
from nucleus.scene import flatten

from .helpers import (
    TEST_CUBOID_ANNOTATIONS,
    TEST_DATASET_3D_NAME,
    TEST_DATASET_ITEMS,
    TEST_LIDAR_ITEMS,
    TEST_LIDAR_SCENES,
    assert_cuboid_annotation_matches_dict,
    assert_partial_equality,
)


@pytest.fixture()
def dataset_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME, is_scene=True)
    yield ds


@pytest.fixture(scope="module")
def dataset_scene_module(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME, is_scene=True)
    yield ds


@pytest.fixture()
def dataset_non_scene(CLIENT):
    ds = CLIENT.create_dataset(
        TEST_DATASET_3D_NAME + " is_scene=False", is_scene=False
    )
    yield ds


@pytest.fixture(scope="module")
@pytest.mark.integration
def scenes_fixture(dataset_scene_module):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene_module.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    expected = {
        "job_id": job.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected, status)

    uploaded_scenes = dataset_scene_module.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )
    assert all(
        u["metadata"] == o.metadata or (not u["metadata"] and not o.metadata)
        for u, o in zip(uploaded_scenes, scenes)
    )
    yield scenes

    for scene in uploaded_scenes:
        dataset_scene_module.delete_scene(scene.reference_id)
    time.sleep(5)
    scenes = dataset_scene_module.scenes
    assert len(scenes) == 0, f"Expected to delete all scenes, got: {scenes}"


def test_frame_add_item():
    frame = Frame()
    frame.add_item(TEST_DATASET_ITEMS[0], "camera")
    frame.add_item(TEST_LIDAR_ITEMS[0], "lidar")

    assert frame.get_sensors() == ["camera", "lidar"]
    for item in frame.get_items():
        assert item in [TEST_DATASET_ITEMS[0], TEST_LIDAR_ITEMS[0]]
    assert frame.get_item("lidar") == TEST_LIDAR_ITEMS[0]
    assert frame.to_payload() == {
        "camera": {
            URL_KEY: TEST_DATASET_ITEMS[0].image_location,
            REFERENCE_ID_KEY: TEST_DATASET_ITEMS[0].reference_id,
            TYPE_KEY: IMAGE_KEY,
            METADATA_KEY: TEST_DATASET_ITEMS[0].metadata or {},
        },
        "lidar": {
            URL_KEY: TEST_LIDAR_ITEMS[0].pointcloud_location,
            REFERENCE_ID_KEY: TEST_LIDAR_ITEMS[0].reference_id,
            TYPE_KEY: POINTCLOUD_KEY,
            METADATA_KEY: TEST_LIDAR_ITEMS[0].metadata or {},
        },
    }


def test_scene_from_json():
    payload = TEST_LIDAR_SCENES
    scene_json = payload[SCENES_KEY][0]
    scene = LidarScene.from_json(scene_json)

    frames = scene_json[FRAMES_KEY]
    camera_item_1 = frames[0]["camera"]
    camera_item = DatasetItem(
        camera_item_1[IMAGE_URL_KEY],
        camera_item_1[REFERENCE_ID_KEY],
        metadata=camera_item_1[METADATA_KEY],
    )
    lidar_item_1 = frames[0]["lidar"]
    lidar_item_f1 = DatasetItem(
        pointcloud_location=lidar_item_1[POINTCLOUD_URL_KEY],
        reference_id=lidar_item_1[REFERENCE_ID_KEY],
        metadata=lidar_item_1[METADATA_KEY],
    )
    expected_items_1 = {
        "camera": camera_item,
        "lidar": lidar_item_f1,
    }
    lidar_item_2 = frames[1]["lidar"]
    lidar_item_f2 = DatasetItem(
        pointcloud_location=lidar_item_2[POINTCLOUD_URL_KEY],
        reference_id=lidar_item_2[REFERENCE_ID_KEY],
        metadata=lidar_item_2[METADATA_KEY],
    )
    expected_items_2 = {
        "lidar": lidar_item_f2,
    }

    expected_frames = [Frame(**expected_items_1), Frame(**expected_items_2)]
    expected_metadata = {"test_meta_field": "test_meta_value"}
    expected_scene = LidarScene(
        scene_json[REFERENCE_ID_KEY],
        expected_frames,
        metadata=expected_metadata,
    )

    assert sorted(
        scene.get_items(), key=lambda item: item.reference_id
    ) == sorted(expected_scene.get_items(), key=lambda item: item.reference_id)
    scene_frames = [frame.to_payload() for frame in scene.get_frames()]
    expected_scene_frames = [
        frame.to_payload() for frame in expected_scene.get_frames()
    ]
    assert scene_frames == expected_scene_frames
    assert set(scene.get_sensors()) == set(expected_scene.get_sensors())
    assert scene.to_payload() == expected_scene.to_payload()


def test_scene_property_methods():
    payload = TEST_LIDAR_SCENES
    scene_json = payload[SCENES_KEY][0]
    scene = LidarScene.from_json(scene_json)

    expected_length = len(scene_json[FRAMES_KEY])
    assert scene.length == expected_length
    sensors = flatten([list(frame.keys()) for frame in scene_json[FRAMES_KEY]])
    expected_num_sensors = len(set(sensors))
    assert scene.num_sensors == expected_num_sensors
    assert scene.info() == {
        REFERENCE_ID_KEY: scene_json[REFERENCE_ID_KEY],
        LENGTH_KEY: expected_length,
        NUM_SENSORS_KEY: expected_num_sensors,
    }


def test_scene_add_item():
    scene_ref_id = "scene_1"
    firstFrame = Frame(lidar=TEST_LIDAR_ITEMS[0])
    scene = LidarScene(scene_ref_id, frames=[firstFrame])
    scene.add_item(0, "camera", TEST_DATASET_ITEMS[0])
    scene.add_item(1, "lidar", TEST_LIDAR_ITEMS[1])
    assert set(scene.get_sensors()) == set(["camera", "lidar"])
    assert scene.get_item(1, "lidar") == TEST_LIDAR_ITEMS[1]
    assert scene.get_items_from_sensor("lidar") == [
        TEST_LIDAR_ITEMS[0],
        TEST_LIDAR_ITEMS[1],
    ]
    assert scene.get_items_from_sensor("camera") == [
        TEST_DATASET_ITEMS[0],
        None,
    ]
    for item in scene.get_items():
        assert item in [
            TEST_DATASET_ITEMS[0],
            TEST_LIDAR_ITEMS[0],
            TEST_LIDAR_ITEMS[1],
        ]

    assert scene.to_payload() == {
        REFERENCE_ID_KEY: scene_ref_id,
        FRAMES_KEY: [
            {
                "camera": {
                    URL_KEY: TEST_DATASET_ITEMS[0].image_location,
                    REFERENCE_ID_KEY: TEST_DATASET_ITEMS[0].reference_id,
                    TYPE_KEY: IMAGE_KEY,
                    METADATA_KEY: TEST_DATASET_ITEMS[0].metadata or {},
                },
                "lidar": {
                    URL_KEY: TEST_LIDAR_ITEMS[0].pointcloud_location,
                    REFERENCE_ID_KEY: TEST_LIDAR_ITEMS[0].reference_id,
                    TYPE_KEY: POINTCLOUD_KEY,
                    METADATA_KEY: TEST_LIDAR_ITEMS[0].metadata or {},
                },
            },
            {
                "lidar": {
                    URL_KEY: TEST_LIDAR_ITEMS[1].pointcloud_location,
                    REFERENCE_ID_KEY: TEST_LIDAR_ITEMS[1].reference_id,
                    TYPE_KEY: POINTCLOUD_KEY,
                    METADATA_KEY: TEST_LIDAR_ITEMS[1].metadata or {},
                }
            },
        ],
    }


def test_scene_add_frame():
    frame_1 = Frame()
    frame_1.add_item(TEST_DATASET_ITEMS[0], "camera")
    frame_1.add_item(TEST_LIDAR_ITEMS[0], "lidar")

    scene_ref_id = "scene_1"
    frames = [frame_1]
    scene = LidarScene(scene_ref_id, frames=frames)

    frame_2 = Frame()
    frame_2.add_item(TEST_LIDAR_ITEMS[1], "lidar")
    scene.add_frame(frame_2, index=1)
    frames.append(frame_2)

    assert scene.length == len(frames)
    assert set(scene.get_sensors()) == set(["camera", "lidar"])
    expected_frame_1 = Frame(
        **{
            "camera": TEST_DATASET_ITEMS[0],
            "lidar": TEST_LIDAR_ITEMS[0],
        },
    )
    assert scene.get_frame(0).to_payload() == expected_frame_1.to_payload()
    expected_frame_2 = Frame(
        **{
            "lidar": TEST_LIDAR_ITEMS[1],
        },
    )
    assert scene.get_frame(1).to_payload() == expected_frame_2.to_payload()
    for item in scene.get_items_from_sensor("lidar"):
        assert item in [TEST_LIDAR_ITEMS[0], TEST_LIDAR_ITEMS[1]]

    assert scene.to_payload() == {
        REFERENCE_ID_KEY: scene_ref_id,
        FRAMES_KEY: [
            {
                "camera": {
                    URL_KEY: TEST_DATASET_ITEMS[0].image_location,
                    REFERENCE_ID_KEY: TEST_DATASET_ITEMS[0].reference_id,
                    TYPE_KEY: IMAGE_KEY,
                    METADATA_KEY: TEST_DATASET_ITEMS[0].metadata or {},
                },
                "lidar": {
                    URL_KEY: TEST_LIDAR_ITEMS[0].pointcloud_location,
                    REFERENCE_ID_KEY: TEST_LIDAR_ITEMS[0].reference_id,
                    TYPE_KEY: POINTCLOUD_KEY,
                    METADATA_KEY: TEST_LIDAR_ITEMS[0].metadata or {},
                },
            },
            {
                "lidar": {
                    URL_KEY: TEST_LIDAR_ITEMS[1].pointcloud_location,
                    REFERENCE_ID_KEY: TEST_LIDAR_ITEMS[1].reference_id,
                    TYPE_KEY: POINTCLOUD_KEY,
                    METADATA_KEY: TEST_LIDAR_ITEMS[1].metadata or {},
                }
            },
        ],
    }


@pytest.mark.integration
def test_scene_upload_and_update(dataset_scene_module, scenes_fixture):
    update_scenes = scenes_fixture
    for scene in update_scenes:
        scene.metadata.update({"new_key": "new_value"})
    job = dataset_scene_module.append(
        update_scenes, update=True, asynchronous=True
    )
    job.sleep_until_complete()
    status = job.status()

    expected = {
        "job_id": job.job_id,
        "status": "Completed",
        "job_type": "uploadLidarScene",
    }
    assert_partial_equality(expected, status)


@pytest.mark.integration
def test_scene_upload_async_item_dataset(dataset_non_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    with pytest.raises(Exception):
        dataset_non_scene.append(scenes, update=update, asynchronous=True)


@pytest.mark.integration
def test_scene_metadata_update(dataset_scene_module, scenes_fixture):
    scene_ref_id = scenes_fixture[0].reference_id
    additional_metadata = {"some_new_key": 123}
    dataset_scene_module.update_scene_metadata(
        {scene_ref_id: additional_metadata}
    )

    expected_new_metadata = {
        **scenes_fixture[0].metadata,
        **additional_metadata,
    }

    updated_scene = dataset_scene_module.get_scene(scene_ref_id)
    actual_metadata = updated_scene.metadata
    assert expected_new_metadata == actual_metadata
