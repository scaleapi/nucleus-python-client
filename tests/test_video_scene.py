import time

import pytest

from nucleus import IMAGE_KEY, UPDATE_KEY, VideoScene
from nucleus.async_job import JobError
from nucleus.constants import (
    FRAME_RATE_KEY,
    FRAMES_KEY,
    LENGTH_KEY,
    METADATA_KEY,
    REFERENCE_ID_KEY,
    SCENES_KEY,
    TYPE_KEY,
    URL_KEY,
    VIDEO_URL_KEY,
)
from tests.helpers import (
    TEST_VIDEO_DATASET_NAME,
    TEST_VIDEO_ITEMS,
    TEST_VIDEO_SCENES,
    TEST_VIDEO_SCENES_INVALID_URLS,
    TEST_VIDEO_SCENES_REPEAT_REF_IDS,
    assert_partial_equality,
)


@pytest.fixture()
def dataset_video_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_VIDEO_DATASET_NAME, is_scene=True)
    yield ds


@pytest.fixture(scope="module")
def dataset_video_module(CLIENT):
    ds = CLIENT.create_dataset(
        TEST_VIDEO_DATASET_NAME + " module scope", is_scene=True
    )
    yield ds


@pytest.fixture(scope="module")
@pytest.mark.integration
def video_scenes(dataset_video_module):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_video_module.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    yield scenes

    uploaded_scenes = dataset_video_module.scenes
    uploaded_scenes.sort(key=lambda x: x["reference_id"])
    assert len(uploaded_scenes) == len(scenes)
    assert all(
        u["reference_id"] == o.reference_id
        for u, o in zip(uploaded_scenes, scenes)
    )

    for scene in uploaded_scenes:
        dataset_video_module.delete_scene(scene.reference_id)
    time.sleep(4)
    assert (
        len(dataset_video_module.scenes) == 0
    ), f"Expected to delete all scenes, got: {dataset_video_module.scenes}"


def test_video_scene_property_methods():
    for scene_json in TEST_VIDEO_SCENES["scenes"]:
        scene = VideoScene.from_json(scene_json)

        expected_frame_rate = scene_json.get("frame_rate", None)
        expected_reference_id = scene_json["reference_id"]
        expected_length = len(scene_json.get("frames", []))
        expected_video_url = scene_json.get("video_url", None)

        info = scene.info()

        assert info[REFERENCE_ID_KEY] == expected_reference_id

        if scene.items and len(scene.items) > 0:
            info[LENGTH_KEY] == expected_length
            assert scene.length == expected_length
            assert info[FRAME_RATE_KEY] == expected_frame_rate
        else:
            assert info[VIDEO_URL_KEY] == expected_video_url


def test_video_scene_add_item():
    scene_ref_id = "scene_1"
    frame_rate = 20
    scene = VideoScene(scene_ref_id, frame_rate)
    scene.add_item(TEST_VIDEO_ITEMS[0])
    scene.add_item(TEST_VIDEO_ITEMS[1], index=1)
    scene.add_item(TEST_VIDEO_ITEMS[2], index=0, update=True)

    assert scene.get_item(0) == TEST_VIDEO_ITEMS[2]
    assert scene.get_item(1) == TEST_VIDEO_ITEMS[1]

    assert scene.get_items() == [
        TEST_VIDEO_ITEMS[2],
        TEST_VIDEO_ITEMS[1],
    ]
    assert scene.to_payload() == {
        REFERENCE_ID_KEY: scene_ref_id,
        FRAME_RATE_KEY: frame_rate,
        FRAMES_KEY: [
            {
                URL_KEY: TEST_VIDEO_ITEMS[2].image_location,
                REFERENCE_ID_KEY: TEST_VIDEO_ITEMS[2].reference_id,
                TYPE_KEY: IMAGE_KEY,
                METADATA_KEY: TEST_VIDEO_ITEMS[2].metadata or {},
            },
            {
                URL_KEY: TEST_VIDEO_ITEMS[1].image_location,
                REFERENCE_ID_KEY: TEST_VIDEO_ITEMS[1].reference_id,
                TYPE_KEY: IMAGE_KEY,
                METADATA_KEY: TEST_VIDEO_ITEMS[1].metadata or {},
            },
        ],
    }


@pytest.mark.integration
def test_video_scene_upload_async(dataset_video_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]

    update = payload[UPDATE_KEY]
    job = dataset_video_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    expected = {
        "job_id": job.job_id,
        "status": "Completed",
        "job_type": "uploadVideoScene",
    }
    assert_partial_equality(expected, status)

    uploaded_scenes = dataset_video_scene.scenes
    uploaded_scenes.sort(key=lambda x: x["reference_id"])
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
@pytest.mark.xfail(reason="SFN doesn't throw on validation error - 05.10.2023")
def test_repeat_refid_video_scene_upload_async(dataset_video_scene):
    payload = TEST_VIDEO_SCENES_REPEAT_REF_IDS
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_video_scene.append(scenes, update=update, asynchronous=True)

    with pytest.raises(JobError):
        job.sleep_until_complete()


@pytest.mark.integration
@pytest.mark.skip(
    reason="This shouldn't be tested in prod - this is a unit test"
)
def test_invalid_url_video_scene_upload_async(dataset_video_scene):
    payload = TEST_VIDEO_SCENES_INVALID_URLS
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_video_scene.append(scenes, update=update, asynchronous=True)
    with pytest.raises(JobError):
        job.sleep_until_complete()


@pytest.mark.integration
def test_video_scene_upload_and_update(dataset_video_module, video_scenes):
    job2 = dataset_video_module.append(
        video_scenes, update=True, asynchronous=True
    )
    job2.sleep_until_complete()
    status2 = job2.status()

    expected2 = {
        "job_id": job2.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected2, status2)


@pytest.mark.integration
def test_video_scene_metadata_update(dataset_video_module, video_scenes):
    scenes = video_scenes
    scene_ref_id = scenes[0].reference_id
    additional_metadata = {"some_new_key": 123}
    dataset_video_module.update_scene_metadata(
        {scene_ref_id: additional_metadata}
    )
    expected_new_metadata = {**scenes[0].metadata, **additional_metadata}

    updated_scene = dataset_video_module.get_scene(scene_ref_id)
    actual_metadata = updated_scene.metadata
    assert expected_new_metadata == actual_metadata


@pytest.mark.integration
def test_video_scene_upload_and_export(dataset_video_module, video_scenes):
    # We re-use the original fixture to avoid long running ingests - this sleep is added
    # to ensure that the scene update has propogated to the readers
    time.sleep(2)
    scenes = dataset_video_module.scenes

    for scene in scenes:
        # Asserting that it doesn't throw -> not the place to test the format
        dataset_video_module.get_scene(scene.reference_id)
