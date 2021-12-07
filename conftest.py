import os

import pytest

import nucleus
from tests.helpers import TEST_DATASET_ITEMS, TEST_DATASET_NAME

assert "NUCLEUS_PYTEST_API_KEY" in os.environ, (
    "You must set the 'NUCLEUS_PYTEST_API_KEY' environment variable to a valid "
    "Nucleus API key to run the test suite"
)

API_KEY = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture(scope="session")
def CLIENT():
    client = nucleus.NucleusClient(API_KEY)
    return client


@pytest.fixture(scope="module")
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds.append(TEST_DATASET_ITEMS)
    yield ds

    CLIENT.delete_dataset(ds.id)


if __name__ == "__main__":
    client = nucleus.NucleusClient(API_KEY)
    # ds = client.create_dataset("Test Dataset With Autotags")
    # ds.append(TEST_DATASET_ITEMS)
    ds = client.get_dataset("ds_c5jwptkgfsqg0cs503z0")
    job = ds.create_image_index()
    job.sleep_until_complete()
    print(ds.id)
