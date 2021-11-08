import pytest
import uuid

from .test_dataset import make_dataset_items
from .helpers import TEST_DATASET_NAME, N_UUID_CHARACTERS, TEST_SLICE_NAME

from nucleus.unit_test import ThresholdComparison

EVAL_FUNCTION_NAME = "IOU"
EVAL_FUNCTION_THRESHOLD = 0.5


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


# TODO: Move unit test to fixture once deletion is implemented
def test_unit_test_creation(CLIENT, dataset):
    # create some dataset_items for the unit test to reference
    items = make_dataset_items()
    dataset.append(items)
    test_name = (
        "unit_test_" + str(uuid.uuid4())[-N_UUID_CHARACTERS:]
    )  # use uuid to make unique
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[items[0].reference_id],
    )

    response = CLIENT.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )
    assert response["unit_test_id"]

    response = CLIENT.create_unit_test_metric(
        unit_test_name=test_name,
        eval_function_name=EVAL_FUNCTION_NAME,
        threshold=EVAL_FUNCTION_THRESHOLD,
        threshold_comparison=ThresholdComparison.GREATER_THAN,
    )
    assert response["unit_test_id"]
    assert response["eval_function_id"]
