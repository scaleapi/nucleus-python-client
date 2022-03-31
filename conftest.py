import os
from typing import TYPE_CHECKING

import pytest

import nucleus
from tests.helpers import TEST_DATASET_ITEMS, TEST_DATASET_NAME

if TYPE_CHECKING:
    from nucleus import NucleusClient

assert "NUCLEUS_PYTEST_API_KEY" in os.environ, (
    "You must set the 'NUCLEUS_PYTEST_API_KEY' environment variable to a valid "
    "Nucleus API key to run the test suite"
)

API_KEY = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture(scope="session")
def CLIENT():
    client = nucleus.NucleusClient(API_KEY)
    return client


@pytest.fixture()
def dataset(CLIENT: "NucleusClient"):
    test_dataset = CLIENT.create_dataset(TEST_DATASET_NAME, is_scene=False)
    test_dataset.append(TEST_DATASET_ITEMS)
    yield test_dataset

    CLIENT.delete_dataset(test_dataset.id)


@pytest.fixture()
def model(CLIENT):
    model = CLIENT.create_model(TEST_DATASET_NAME, "fake_reference_id")
    yield model
    CLIENT.delete_model(model.id)


if __name__ == "__main__":
    client = nucleus.NucleusClient(API_KEY)
    # ds = client.create_dataset("Test Dataset With Autotags")
    # ds.append(TEST_DATASET_ITEMS)
    ds = client.get_dataset("ds_c5jwptkgfsqg0cs503z0")
    job = ds.create_image_index()
    job.sleep_until_complete()
    print(ds.id)
