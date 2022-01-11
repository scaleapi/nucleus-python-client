import os

import pytest

from tests.helpers import get_uuid

os.environ["NUCLEUS_API_KEY"] = os.environ["NUCLEUS_PYTEST_API_KEY"]


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
