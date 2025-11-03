import os

import pytest
from click.testing import CliRunner

from tests.helpers import create_box_annotations, create_predictions, get_uuid
from tests.test_dataset import make_dataset_items

os.environ["NUCLEUS_API_KEY"] = os.environ.get("NUCLEUS_PYTEST_API_KEY") if "NUCLEUS_PYTEST_API_KEY" in os.environ else None
os.environ["NUCLEUS_LIMITED_ACCESS_KEY"] = os.environ.get("NUCLEUS_PYTEST_LIMITED_ACCESS_KEY") if "NUCLEUS_PYTEST_LIMITED_ACCESS_KEY" in os.environ else None


@pytest.fixture
def runner():
    yield CliRunner()


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


@pytest.fixture(scope="function")
def function_scope_dataset(CLIENT):
    dataset = CLIENT.create_dataset(f"[PyTest] Dataset {get_uuid()}")
    yield dataset


@pytest.fixture(scope="module")
def module_scope_models(CLIENT):
    models = []
    for i in range(3):
        model_name = "[PyTest] Model {i}"
        model_ref = f"pytest_model_{i}_{get_uuid()}"
        models.append(CLIENT.create_model(model_name, model_ref))
    yield models

    for model in models:
        CLIENT.delete_model(model.id)


@pytest.fixture(scope="module")
def populated_dataset(module_scope_datasets):
    yield module_scope_datasets[0]


@pytest.fixture(scope="module")
def populated_scene_dataset(module_scope_scene_datasets):
    yield module_scope_scene_datasets[0]


@pytest.fixture(scope="module")
def model(module_scope_models):
    yield module_scope_models[0]


@pytest.fixture(scope="module")
def dataset_items(populated_dataset):
    items = make_dataset_items()
    populated_dataset.append(items)
    yield items


@pytest.fixture(scope="module")
def slice_items(dataset_items):
    yield dataset_items[:2]


@pytest.fixture(scope="module")
def test_slice(populated_dataset, slice_items):
    slice_name = "[PyTest] CLI Slice"
    slc = populated_dataset.create_slice(
        name=slice_name,
        reference_ids=[item.reference_id for item in slice_items],
    )
    yield slc


@pytest.fixture(scope="module")
def scenes(populated_dataset):
    items = make_dataset_items()
    populated_dataset.append(items)
    yield items


@pytest.fixture(scope="module")
def slice_scenes(scenes):
    yield scenes[:2]


@pytest.fixture(scope="module")
def test_scene_slice(populated_scene_dataset, slice_scenes):
    slice_name = "[PyTest] CLI Scene Slice"
    slc = populated_scene_dataset.create_slice(
        name=slice_name,
        reference_ids=[scene.reference_id for scene in slice_scenes],
    )
    yield slc


@pytest.fixture(scope="module")
def annotations(populated_dataset, slice_items):
    annotations = create_box_annotations(populated_dataset, slice_items)
    yield annotations


@pytest.fixture(scope="module")
def predictions(model, populated_dataset, annotations):
    predictions = create_predictions(populated_dataset, model, annotations)
    yield predictions


@pytest.fixture(scope="module")
@pytest.mark.usefixtures(
    "annotations"
)  # Unit test needs to have annotations in the slice
def scenario_test(CLIENT, test_slice, annotations, predictions):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[CLIENT.validate.eval_functions.bbox_recall],
    )
    yield scenario_test

    CLIENT.validate.delete_scenario_test(scenario_test.id)
