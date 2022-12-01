import time
from copy import deepcopy

import pytest

from nucleus.annotation import BoxAnnotation
from nucleus.scene import VideoScene
from tests.helpers import (
    TEST_MODEL_NAME,
    TEST_SCENE_BOX_ANNS_WITH_TRACK,
    TEST_SCENE_BOX_PREDS_WITH_TRACK,
    TEST_VIDEO_DATASET_NAME,
    TEST_VIDEO_SCENES,
)


@pytest.fixture(scope="session")
def dataset_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_VIDEO_DATASET_NAME, is_scene=True)

    # Upload scenes only
    scenes = []
    for scene in TEST_VIDEO_SCENES["scenes"]:
        scenes.append(VideoScene.from_json(scene, CLIENT))
    job = ds.append(
        scenes,
        asynchronous=True,
        update=TEST_VIDEO_SCENES["update"],
    )
    job.sleep_until_complete()

    yield ds

    # Delete dataset after all tests finish
    CLIENT.delete_dataset(ds.id)


def test_create_gt_with_tracks(dataset_scene):
    # Arrange
    expected_track_reference_ids = [
        ann["track_reference_id"] for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ]
    annotations = [
        BoxAnnotation.from_json(ann) for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ]

    # Act
    dataset_scene.annotate(
        annotations=annotations,
        update=False,
        asynchronous=False,
    )

    # Assert
    assert set(
        [track.reference_id for track in dataset_scene.tracks]
    ).issubset(expected_track_reference_ids)

    # Cleanup
    job = dataset_scene.delete_annotations(
        reference_ids=[ann.reference_id for ann in annotations]
    )
    job.sleep_until_complete()
    assert job.status()["status"] == "Completed"
    dataset_scene.delete_tracks(expected_track_reference_ids)


def test_create_mp_with_tracks(CLIENT, dataset_scene):
    # Arrange
    expected_track_reference_ids = [
        ann["track_reference_id"] for ann in TEST_SCENE_BOX_PREDS_WITH_TRACK
    ]
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)

    # Act
    dataset_scene.upload_predictions(
        model=model,
        predictions=[
            BoxAnnotation.from_json(ann)
            for ann in TEST_SCENE_BOX_PREDS_WITH_TRACK
        ],
        update=False,
        asynchronous=False,
    )

    # Assert
    assert set(
        [track.reference_id for track in dataset_scene.tracks]
    ).issubset(expected_track_reference_ids)

    # Cleanup
    assert CLIENT.delete_model(model.id) == {}
    dataset_scene.delete_tracks(expected_track_reference_ids)


def test_update_tracks_metadata(dataset_scene):
    # Arrange
    annotations = [
        BoxAnnotation.from_json(ann) for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ][:2]
    expected_track_reference_ids = [
        ann.track_reference_id for ann in annotations
    ]
    dataset_scene.annotate(
        annotations=annotations,
        update=False,
        asynchronous=False,
    )
    new_metadata_1 = {
        "is_completely_new": True,
    }
    new_metadata_2 = {
        "is_new_key": "value",
    }
    [original_track] = dataset_scene.tracks

    # Act
    try:
        original_track.update(new_metadata_1, overwrite_metadata=True)
        [track_update_1] = dataset_scene.tracks
        # Have to copy because track_update_1 gets mutated in place
        deepcopy(track_update_1).update(
            new_metadata_2, overwrite_metadata=False
        )
        [track_update_2] = dataset_scene.tracks
    except Exception as e:
        assert False, f"Updating tracks raised an exception: {e}"

    # Assert
    assert original_track.metadata == new_metadata_1
    assert track_update_1.metadata == new_metadata_1
    assert track_update_2.metadata["is_new_key"] == "value"

    # Cleanup
    job = dataset_scene.delete_annotations(
        reference_ids=[ann.reference_id for ann in annotations]
    )
    job.sleep_until_complete()
    assert job.status()["status"] == "Completed"
    dataset_scene.delete_tracks(expected_track_reference_ids)
