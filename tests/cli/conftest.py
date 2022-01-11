import os

import pytest
from click.testing import CliRunner

from tests.helpers import create_box_annotations, create_predictions, get_uuid
from tests.test_dataset import make_dataset_items

os.environ["NUCLEUS_API_KEY"] = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture
def runner():
    yield CliRunner()


@pytest.fixture(scope="module")
def cli_datasets(CLIENT):
    test_datasets = []
    for i in range(3):
        dataset_name = f"[PyTest] CLI {i} {get_uuid()}"
        test_datasets.append(
            CLIENT.create_dataset(dataset_name, is_scene=False)
        )
    yield test_datasets

    for test_dataset in test_datasets:
        CLIENT.delete_dataset(test_dataset.id)


@pytest.fixture(scope="module")
def cli_models(CLIENT):
    models = []
    for i in range(3):
        model_name = "[PyTest] Model {i}"
        model_ref = f"pytest_model_{i}_{get_uuid()}"
        models.append(CLIENT.create_model(model_name, model_ref))
    yield models

    for model in models:
        CLIENT.delete_model(model.id)


@pytest.fixture(scope="module")
def populated_dataset(cli_datasets):
    yield cli_datasets[0]


@pytest.fixture(scope="module")
def model(cli_models):
    yield cli_models[0]


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
def unit_test(CLIENT, test_slice, annotations, predictions):
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_criteria=[CLIENT.modelci.eval_functions.bbox_recall > 0.5],
    )
    yield unit_test

    CLIENT.modelci.delete_unit_test(unit_test.id)
