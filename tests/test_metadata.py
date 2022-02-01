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


def test_fetch_scene_metadata(dataset_scene, uploaded_scenes):

   assert uploaded_scenes == 1

