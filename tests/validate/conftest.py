import time

import pytest

from tests.helpers import (
    TEST_MODEL_NAME,
    TEST_SLICE_NAME,
    create_box_annotations,
    create_predictions,
    get_uuid,
)
from tests.test_dataset import make_dataset_items, make_scenes


@pytest.fixture(scope="module")
def validate_dataset(CLIENT):
    """SHOULD NOT BE MUTATED IN TESTS. This dataset lives for the whole test module scope."""
    ds = CLIENT.create_dataset("[Test Model CI] Dataset", is_scene=False)
    yield ds


@pytest.fixture(scope="module")
def dataset_items(validate_dataset):
    items = make_dataset_items()
    validate_dataset.append(items)
    yield items


@pytest.fixture(scope="module")
def slice_items(dataset_items):
    yield dataset_items[:2]


@pytest.fixture(scope="module")
def test_slice(validate_dataset, slice_items):
    slc = validate_dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in slice_items],
    )
    yield slc


@pytest.fixture(scope="module")
def module_scope_datasets(CLIENT):
    test_datasets = []
    for i in range(3):
        dataset_name = f"[PyTest] CLI {i} {get_uuid()}"
        test_datasets.append(
            CLIENT.create_dataset(dataset_name, is_scene=False)
        )
    yield test_datasets


@pytest.fixture(scope="module")
def module_scope_scene_datasets(CLIENT):
    test_scene_datasets = []
    for i in range(3):
        dataset_name = f"[PyTest] CLI {i} {get_uuid()} (Scene)"
        test_scene_datasets.append(
            CLIENT.create_dataset(dataset_name, is_scene=True)
        )
    yield test_scene_datasets


@pytest.fixture(scope="module")
def populated_scene_dataset(module_scope_scene_datasets):
    yield module_scope_scene_datasets[0]


@pytest.fixture(scope="module")
def slice_scenes():
    scenes = make_scenes()[:1]
    yield scenes


@pytest.fixture(scope="module")
def scenes(populated_scene_dataset, slice_scenes):
    job = populated_scene_dataset.append(slice_scenes, asynchronous=True)
    job.sleep_until_complete()
    yield slice_scenes


@pytest.fixture(scope="module")
def test_scene_slice(populated_scene_dataset, scenes):
    slice_name = "[PyTest] CLI Scene Slice"
    slc = populated_scene_dataset.create_slice(
        name=slice_name,
        reference_ids=[scene.reference_id for scene in scenes],
    )
    yield slc


@pytest.fixture(scope="module")
def model(CLIENT):
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
    yield model

    CLIENT.delete_model(model.id)


@pytest.fixture(scope="module")
def annotations(validate_dataset, slice_items):
    annotations = create_box_annotations(validate_dataset, slice_items)
    yield annotations


@pytest.fixture(scope="module")
def predictions(model, validate_dataset, annotations):
    predictions = create_predictions(validate_dataset, model, annotations)
    yield predictions


@pytest.fixture(scope="module")
@pytest.mark.usefixtures(
    "annotations"
)  # Scenario test needs to have annotations in the slice
def scenario_test(CLIENT, test_slice):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[CLIENT.validate.eval_functions.bbox_recall],
    )
    yield scenario_test

    CLIENT.validate.delete_scenario_test(scenario_test.id)
