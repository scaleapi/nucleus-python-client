import pytest
import time

from tests.helpers import TEST_MODEL_NAME


@pytest.fixture()
def model(CLIENT):
    model_reference = "model_" + str(time.time())
    model = CLIENT.add_model(TEST_MODEL_NAME, model_reference)
    yield model

    CLIENT.delete_model(model.id)
