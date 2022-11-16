import pytest

from nucleus.annotation import BoxAnnotation
from tests.helpers import (
    TEST_MODEL_NAME,
    TEST_SCENE_BOX_ANNS_WITH_TRACK,
    TEST_SCENE_BOX_PREDS_WITH_TRACK,
)


def test_create_gt_with_tracks(populated_scene_dataset):
    # Arrange
    expected_track_reference_ids = [
        ann["track_reference_id"] for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ]
    annotations = [
        BoxAnnotation.from_json(ann) for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ]
    # Act
    populated_scene_dataset.annotate(
        annotations=[annotations],
        update=False,
        asynchronous=False,
    )
    # Assert
    assert set(
        [track.reference_id for track in populated_scene_dataset.tracks]
    ).issubset(expected_track_reference_ids)
    # Cleanup
    job = populated_scene_dataset.delete_annotations(
        reference_ids=[ann.reference_id for ann in annotations]
    )
    job.sleep_until_complete()
    assert job.status()["status"] == "Completed"
    populated_scene_dataset.delete_tracks(expected_track_reference_ids)


def test_create_mp_with_tracks(CLIENT, populated_scene_dataset):
    # Arrange
    expected_track_reference_ids = [
        ann["track_reference_id"] for ann in TEST_SCENE_BOX_PREDS_WITH_TRACK
    ]
    model_reference = "model_test_create_mp_with_tracks"
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
    # Act
    populated_scene_dataset.upload_predictions(
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
        [track.reference_id for track in populated_scene_dataset.tracks]
    ).issubset(expected_track_reference_ids)
    # Cleanup
    assert CLIENT.delete_model(model.id) == {}
    populated_scene_dataset.delete_tracks(expected_track_reference_ids)


def test_update_tracks_metadata(populated_scene_dataset):
    # Arrange
    annotations = [
        BoxAnnotation.from_json(ann) for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ][:2]
    expected_track_reference_ids = [
        ann.track_reference_id for ann in annotations
    ]
    # Act
    populated_scene_dataset.annotate(
        annotations=[annotations],
        update=False,
        asynchronous=False,
    )
    new_metadata_1 = {
        "is_completely_new": True,
    }
    new_metadata_2 = {
        "is_new_key": "value",
    }
    [track_1, track_2] = populated_scene_dataset.tracks
    # Assert
    try:
        track_1.update(new_metadata_1, overwrite_metadata=True)
        track_2.update(new_metadata_2, overwrite_metadata=False)
    except Exception as e:
        assert False, f"Updating tracks raised an exception: {e}"
    [new_track_1, new_track_2] = populated_scene_dataset.tracks
    assert new_track_1.metadata == new_metadata_1
    assert new_track_2.metadata["is_new_key"] == "value"
    # Cleanup
    job = populated_scene_dataset.delete_annotations(
        reference_ids=[ann.reference_id for ann in annotations]
    )
    job.sleep_until_complete()
    assert job.status()["status"] == "Completed"
    populated_scene_dataset.delete_tracks(expected_track_reference_ids)
