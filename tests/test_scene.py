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
)


@pytest.fixture()
def dataset_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME, is_scene=True)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


@pytest.fixture()
def dataset_item(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME, is_scene=False)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


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
    expected_scene = LidarScene(
        scene_json[REFERENCE_ID_KEY], expected_frames, metadata={}
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
    scene = LidarScene(scene_ref_id)
    scene.add_item(0, "camera", TEST_DATASET_ITEMS[0])
    scene.add_item(0, "lidar", TEST_LIDAR_ITEMS[0])
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


@pytest.mark.skip("Deactivated sync upload for scenes")
def test_scene_upload_sync(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    response = dataset_scene.append(scenes, update=update)

    first_scene = dataset_scene.get_scene(scenes[0].reference_id)

    assert first_scene == scenes[0]
    first_scene_modified = copy.deepcopy(first_scene)
    first_scene_modified.reference_id = "WRONG!"
    assert first_scene_modified != scenes[0]

    assert response["dataset_id"] == dataset_scene.id
    assert response["new_scenes"] == len(scenes)

    uploaded_scenes = dataset_scene.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )
    assert all(
        u["metadata"] == o.metadata or (not u["metadata"] and not o.metadata)
        for u, o in zip(uploaded_scenes, scenes)
    )


@pytest.mark.skip("Deactivated sync upload for scenes")
@pytest.mark.integration
def test_scene_and_cuboid_upload_sync(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    response = dataset_scene.append(scenes, update=update)

    assert response["dataset_id"] == dataset_scene.id
    assert response["new_scenes"] == len(scenes)

    uploaded_scenes = dataset_scene.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )
    assert all(
        u["metadata"] == o.metadata or (not u["metadata"] and not o.metadata)
        for u, o in zip(uploaded_scenes, scenes)
    )

    lidar_item_ref = payload[SCENES_KEY][0][FRAMES_KEY][0]["lidar"][
        REFERENCE_ID_KEY
    ]
    TEST_CUBOID_ANNOTATIONS[0][REFERENCE_ID_KEY] = lidar_item_ref

    annotations = [CuboidAnnotation.from_json(TEST_CUBOID_ANNOTATIONS[0])]
    response = dataset_scene.annotate(annotations)

    assert response["dataset_id"] == dataset_scene.id
    assert response["annotations_processed"] == len(annotations)
    assert response["annotations_ignored"] == 0

    response_annotations = dataset_scene.refloc(lidar_item_ref)[
        ANNOTATIONS_KEY
    ]["cuboid"]
    assert len(response_annotations) == 1
    assert_cuboid_annotation_matches_dict(
        response_annotations[0], TEST_CUBOID_ANNOTATIONS[0]
    )


@pytest.mark.integration
def test_scene_upload_async(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    assert status == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "scene_upload_progress": {
                "errors": [],
                "dataset_id": dataset_scene.id,
                "new_scenes": len(scenes),
                "ignored_scenes": 0,
                "scenes_errored": 0,
                "updated_scenes": 0,
            }
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }

    uploaded_scenes = dataset_scene.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )
    assert all(
        u["metadata"] == o.metadata or (not u["metadata"] and not o.metadata)
        for u, o in zip(uploaded_scenes, scenes)
    )


@pytest.mark.integration
def test_scene_upload_and_update(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    assert status == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "scene_upload_progress": {
                "errors": [],
                "dataset_id": dataset_scene.id,
                "new_scenes": len(scenes),
                "ignored_scenes": 0,
                "scenes_errored": 0,
                "updated_scenes": 0,
            }
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }

    uploaded_scenes = dataset_scene.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )
    assert all(
        u["metadata"] == o.metadata or (not u["metadata"] and not o.metadata)
        for u, o in zip(uploaded_scenes, scenes)
    )

    job2 = dataset_scene.append(scenes, update=True, asynchronous=True)
    job2.sleep_until_complete()
    status2 = job2.status()

    assert status2 == {
        "job_id": job2.job_id,
        "status": "Completed",
        "message": {
            "scene_upload_progress": {
                "errors": [],
                "dataset_id": dataset_scene.id,
                "new_scenes": 0,
                "ignored_scenes": 0,
                "scenes_errored": 0,
                "updated_scenes": len(scenes),
            }
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }


@pytest.mark.integration
def test_scene_deletion(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()

    uploaded_scenes = dataset_scene.scenes
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )

    for scene in uploaded_scenes:
        dataset_scene.delete_scene(scene.reference_id)
    time.sleep(1)
    scenes = dataset_scene.scenes
    assert len(scenes) == 0, f"Expected to delete all scenes, got: {scenes}"


@pytest.mark.integration
def test_scene_upload_async_item_dataset(dataset_item):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    with pytest.raises(Exception):
        dataset_item.append(scenes, update=update, asynchronous=True)
