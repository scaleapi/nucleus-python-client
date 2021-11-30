import time

import pytest

from tests.helpers import TEST_MODEL_NAME, TEST_SLICE_NAME, get_uuid
from tests.test_dataset import make_dataset_items


@pytest.fixture()
def model(CLIENT):
    model_reference = "model_" + str(time.time())
    model = CLIENT.add_model(TEST_MODEL_NAME, model_reference)
    yield model

    CLIENT.delete_model(model.id)


@pytest.fixture()
def unit_test(CLIENT, dataset):
    items = make_dataset_items()
    dataset.append(items)
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[items[0].reference_id],
    )
    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )
    yield unit_test

    CLIENT.modelci.delete_unit_test(unit_test.id)
