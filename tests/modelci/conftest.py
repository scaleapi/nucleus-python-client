import os
import pytest
import time

from nucleus.modelci import ModelCIClient

from tests.helpers import TEST_DATASET_NAME, TEST_MODEL_NAME

assert "NUCLEUS_PYTEST_API_KEY" in os.environ, (
    "You must set the 'NUCLEUS_PYTEST_API_KEY' environment variable to a valid "
    "Nucleus API key to run the test suite"
)

API_KEY = os.environ["NUCLEUS_PYTEST_API_KEY"]


@pytest.fixture(scope="session")
def MODELCI_CLIENT():
    client = ModelCIClient(API_KEY)
    return client


@pytest.fixture()
def dataset(MODELCI_CLIENT):
    ds = MODELCI_CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = MODELCI_CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


@pytest.fixture()
def model(MODELCI_CLIENT):
    model_reference = "model_" + str(time.time())
    model = MODELCI_CLIENT.add_model(TEST_MODEL_NAME, model_reference)
    yield model

    MODELCI_CLIENT.delete_model(model.id)
