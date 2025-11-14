import os
import random
import sys
from typing import TYPE_CHECKING

import pytest

import nucleus
from tests.helpers import TEST_DATASET_ITEMS, TEST_DATASET_NAME

if TYPE_CHECKING:
    from nucleus import NucleusClient

assert "NUCLEUS_PYTEST_API_KEY" in os.environ or "NUCLEUS_PYTEST_LIMITED_ACCESS_KEY" in os.environ, (
    "You must set at least one of 'NUCLEUS_PYTEST_API_KEY' or "
    "'NUCLEUS_PYTEST_LIMITED_ACCESS_KEY' environment variables to run the test suite"
)

API_KEY = os.environ.get("NUCLEUS_PYTEST_API_KEY") if "NUCLEUS_PYTEST_API_KEY" in os.environ else None
LIMITED_ACCESS_KEY = os.environ.get("NUCLEUS_PYTEST_LIMITED_ACCESS_KEY") if "NUCLEUS_PYTEST_LIMITED_ACCESS_KEY" in os.environ else None


@pytest.fixture(scope="session")
def CLIENT():
    # if API_KEY and LIMITED_ACCESS_KEY:
    #     return nucleus.NucleusClient(api_key=API_KEY, limited_access_key=LIMITED_ACCESS_KEY)
    if API_KEY:
        return nucleus.NucleusClient(api_key=API_KEY)
    # LIMITED_ACCESS_KEY only
    return nucleus.NucleusClient(limited_access_key=LIMITED_ACCESS_KEY)


@pytest.fixture()
def dataset(CLIENT: "NucleusClient"):
    test_dataset = CLIENT.create_dataset(TEST_DATASET_NAME, is_scene=False)
    test_dataset.append(TEST_DATASET_ITEMS)
    yield test_dataset


@pytest.fixture()
def model(CLIENT):
    # Randomly generate an integer between 0 and maximum integer so reference ids
    # do not collide during parallel test rusn.
    random_postfix = str(random.randint(0, sys.maxsize))
    model = CLIENT.create_model(
        TEST_DATASET_NAME, "fake_reference_id_" + random_postfix
    )
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
