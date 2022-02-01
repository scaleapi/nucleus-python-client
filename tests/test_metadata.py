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

RAW_DATASET_ITEMS_META = {}
for item in TEST_LIDAR_SCENES[SCENES_KEY][0][FRAMES_KEY]:
    for v in item.values():
        RAW_DATASET_ITEMS_META[v['reference_id']] = v['metadata']


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
    """
    [ScenesListEntry(id='scn_c7ws4cskr9kcyb1embjg', reference_id='scene_1', type='lidar', metadata=None)] == 1
    """
    mm = dataset_scene.metadata_manager()
    assert len(uploaded_scenes) == 1
    sceneMeta = mm.load_scenes(uploaded_scenes[0].reference_id)
    assert len(sceneMeta) == 1
    assert len(sceneMeta[0].dataset_items) == 0
    assert sceneMeta[0].reference_id == uploaded_scenes[0].reference_id
    assert sceneMeta[0].metadata == uploaded_scenes[0].metadata


def test_fetch_scene_metadata_with_item(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    """
    [ScenesListEntry(id='scn_c7ws4cskr9kcyb1embjg', reference_id='scene_1', type='lidar', metadata=None)] == 1
    """
    mm = dataset_scene.metadata_manager()
    assert len(uploaded_scenes) == 1
    sceneMeta = mm.load_scenes(uploaded_scenes[0].reference_id, True)
    assert len(sceneMeta) == 1
    assert sceneMeta[0].reference_id == uploaded_scenes[0].reference_id
    assert sceneMeta[0].metadata == uploaded_scenes[0].metadata

    assert len(sceneMeta[0].dataset_items) == 3
    for meta_items in sceneMeta[0].dataset_items:
        assert meta_items.metadata == RAW_DATASET_ITEMS_META[meta_items.reference_id]


def test_fetch_dataset_items(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    """
    [ScenesListEntry(id='scn_c7ws4cskr9kcyb1embjg', reference_id='scene_1', type='lidar', metadata=None)] == 1
    """
    mm = dataset_scene.metadata_manager()
    itemsMeta = mm.load_dataset_items(list(RAW_DATASET_ITEMS_META.keys()))
    assert len(itemsMeta) == len(RAW_DATASET_ITEMS_META.keys())
    for item in itemsMeta:
        assert item.metadata == raw_meta[item.reference_id]


def test_update_scene_metadata(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    sceneMeta = mm.load_scenes(uploaded_scenes[0].reference_id, False)

    for item in sceneMeta:
        item.metadata['new_meta_field'] = '123'

    mm.update(sceneMeta)

    sceneMetaUpdated = mm.load_scenes(uploaded_scenes[0].reference_id, False)
    assert sceneMetaUpdated[0].metadata == {
        "new_meta_field": "123", "meta_int": 123, "meta_str": "foo"
    }


def test_update_items_metadata(dataset_scene: Dataset, uploaded_scenes: List[ScenesListEntry]):
    mm = dataset_scene.metadata_manager()
    itemsMeta = mm.load_dataset_items(list(RAW_DATASET_ITEMS_META.keys()))

    for i, item in enumerate(itemsMeta):
        item.metadata['new_meta_field'] = i

    mm.update(itemsMeta)

    itemsMetaUpdated = mm.load_dataset_items(list(RAW_DATASET_ITEMS_META.keys()))
    for i, updated_items in enumerate(itemsMetaUpdated):
        expected_new_meta = dict(
            list(updated_items.metadata.items()) +
            list(RAW_DATASET_ITEMS_META[updated_items.reference_id]))
        assert updated_items.metadata == expected_new_meta


if __name__ == '__main__':
    raw_meta = {item.keys()[0]["reference_id"]: item.keys()[0]["metadata"] for item in
                TEST_LIDAR_SCENES[SCENES_KEY][FRAMES_KEY]}

    print(raw_meta)
