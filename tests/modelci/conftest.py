import time

import pytest

from tests.helpers import TEST_MODEL_NAME, TEST_SLICE_NAME, get_uuid
from tests.test_dataset import make_dataset_items


@pytest.fixture()
def model(CLIENT):
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
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
        slice_id=slc.id,
        evaluation_criteria=[CLIENT.modelci.eval_functions.bbox_recall > 0.5],
    )
    yield unit_test

    CLIENT.modelci.delete_unit_test(unit_test.id)


@pytest.fixture()
def test_slice(CLIENT, dataset):
    items = make_dataset_items()
    dataset.append(items)
    slice_name = TEST_SLICE_NAME + f"_{get_uuid()}"
    slc = dataset.create_slice(
        name=slice_name,
        reference_ids=[items[0].reference_id],
    )
    yield slc

    CLIENT.delete_slice(slc.id)
