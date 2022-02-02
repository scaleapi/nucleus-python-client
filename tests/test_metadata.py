from copy import deepcopy
from typing import List

import pytest

from nucleus import Dataset, LidarScene
from nucleus.constants import (
    FRAMES_KEY, SCENES_KEY,
    UPDATE_KEY,
)
from nucleus.data_transfer_object.scenes_list import ScenesListEntry
from .helpers import (
    TEST_DATASET_3D_NAME,
    TEST_LIDAR_SCENES,
)

EXPECTED_DATASET_ITEMS_META = {}
for item in TEST_LIDAR_SCENES[SCENES_KEY][0][FRAMES_KEY]:
    for v in item.values():
        EXPECTED_DATASET_ITEMS_META[v["reference_id"]] = v["metadata"]


@pytest.fixture(scope="module")
def dataset_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_3D_NAME, is_scene=True)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


@pytest.fixture(scope="module")
def uploaded_scenes(dataset_scene):
    payload = TEST_LIDAR_SCENES
    scenes = [
        LidarScene.from_json(scene_json) for scene_json in payload[SCENES_KEY]
    ]
    update = payload[UPDATE_KEY]

    job = dataset_scene.append(scenes, update=update, asynchronous=True)
    job.sleep_until_complete()

    uploaded_scenes = dataset_scene.scenes
    yield uploaded_scenes


def test_fetch_scene_metadata(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    assert len(uploaded_scenes) == 1
    scene_meta = mm.load_scenes(uploaded_scenes[0].reference_id)
    assert len(scene_meta) == 1
    assert len(scene_meta[0].dataset_items) == 0
    assert scene_meta[0].reference_id == uploaded_scenes[0].reference_id
    assert scene_meta[0].metadata == uploaded_scenes[0].metadata


def test_fetch_scene_metadata_with_item(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    assert len(uploaded_scenes) == 1
    scene_meta = mm.load_scenes(uploaded_scenes[0].reference_id, True)
    assert len(scene_meta) == 1
    assert scene_meta[0].reference_id == uploaded_scenes[0].reference_id
    assert scene_meta[0].metadata == uploaded_scenes[0].metadata

    assert len(scene_meta[0].dataset_items) == 3
    for item in scene_meta[0].dataset_items:
        assert item.metadata == EXPECTED_DATASET_ITEMS_META[item.reference_id]


def test_fetch_dataset_items(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    items_meta = mm.load_dataset_items(list(EXPECTED_DATASET_ITEMS_META.keys()))
    assert len(items_meta) == len(EXPECTED_DATASET_ITEMS_META.keys())
    for item in items_meta:
        assert item.metadata == EXPECTED_DATASET_ITEMS_META[item.reference_id]


def test_update_scene_metadata(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    scene_meta = mm.load_scenes(uploaded_scenes[0].reference_id, False)

    for item in scene_meta:
        item.metadata["new_meta_field"] = "123"

    mm.update(scene_meta)

    updated_scene_meta = mm.load_scenes(uploaded_scenes[0].reference_id, False)
    expected_scene_metadata = {"new_meta_field": "123", "meta_int": 123, "meta_str": "foo"}
    assert updated_scene_meta[0].metadata == expected_scene_metadata


def test_update_items_metadata(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    items_meta = mm.load_dataset_items(list(EXPECTED_DATASET_ITEMS_META.keys()))

    expected_updated_items = {}
    for i, item in enumerate(items_meta):
        item.metadata["new_meta_field"] = i
        expected_updated_items[item.reference_id] = deepcopy(item.metadata)
        expected_updated_items[item.reference_id]["new_meta_field"] = i

    mm.update(items_meta)

    updated_items = mm.load_dataset_items(list(EXPECTED_DATASET_ITEMS_META.keys()))

    updated_items_meta = [item.metadata for item in updated_items]
    for item in updated_items:
        assert item.metadata == expected_updated_items[item.reference_id]

