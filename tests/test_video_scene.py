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
    UPLOAD_TO_SCALE_KEY,
    URL_KEY,
    VIDEO_URL_KEY,
)
from tests.helpers import (
    TEST_VIDEO_ITEMS,
    TEST_VIDEO_SCENES,
    TEST_VIDEO_SCENES_INVALID_URLS,
    TEST_VIDEO_SCENES_REPEAT_REF_IDS,
    assert_partial_equality,
)


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
        UPLOAD_TO_SCALE_KEY: True,
    }


@pytest.mark.integration
def test_video_scene_upload_async(dataset_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]

    update = payload[UPDATE_KEY]
    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    del status["job_creation_time"]  # HACK: too flaky to try syncing
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
        "completed_steps": len(scenes),
        "total_steps": len(scenes),
        "job_last_known_status": "Completed",
        "job_type": "uploadVideoScene",
    }

    uploaded_scenes = dataset_scene.scenes
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
def test_repeat_refid_video_scene_upload_async(dataset_scene):
    payload = TEST_VIDEO_SCENES_REPEAT_REF_IDS
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_scene.append(scenes, update=update, asynchronous=True)

    with pytest.raises(JobError):
        job.sleep_until_complete()


@pytest.mark.integration
def test_invalid_url_video_scene_upload_async(dataset_scene):
    payload = TEST_VIDEO_SCENES_INVALID_URLS
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    with pytest.raises(JobError):
        job.sleep_until_complete()


@pytest.mark.integration
def test_video_scene_upload_and_update(dataset_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()
    status = job.status()

    del status["job_creation_time"]  # HACK: too flaky to try syncing
    expected = {
        "job_id": job.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected, status)

    uploaded_scenes = dataset_scene.scenes
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

    job2 = dataset_scene.append(scenes, update=True, asynchronous=True)
    job2.sleep_until_complete()
    status2 = job2.status()

    del status2["job_creation_time"]  # HACK: too flaky to try syncing
    expected = {
        "job_id": job2.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected, status)


@pytest.mark.integration
def test_video_scene_deletion(dataset_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()

    uploaded_scenes = dataset_scene.scenes
    uploaded_scenes.sort(key=lambda x: x["reference_id"])
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
def test_video_scene_metadata_update(dataset_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()

    scene_ref_id = scenes[0].reference_id
    additional_metadata = {"some_new_key": 123}
    dataset_scene.update_scene_metadata({scene_ref_id: additional_metadata})
    expected_new_metadata = {**scenes[0].metadata, **additional_metadata}

    updated_scene = dataset_scene.get_scene(scene_ref_id)
    actual_metadata = updated_scene.metadata
    assert expected_new_metadata == actual_metadata


@pytest.mark.integration
def test_video_scene_upload_and_export(dataset_scene):
    payload = TEST_VIDEO_SCENES
    scenes = [
        VideoScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]
    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()

    for scene in scenes:
        get_scene_result = dataset_scene.get_scene(scene.reference_id)
        assert scene.to_payload() == get_scene_result.to_payload()
