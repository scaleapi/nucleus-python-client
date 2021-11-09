import uuid

from ..test_dataset import make_dataset_items
from ..helpers import (
    N_UUID_CHARACTERS,
    TEST_SLICE_NAME,
    EVAL_FUNCTION_NAME,
    EVAL_FUNCTION_THRESHOLD,
)

from nucleus.modelci.unit_test import ThresholdComparison, UnitTest


# TODO: Move unit test to fixture once deletion is implemented
def test_unit_test_creation(MODELCI_CLIENT, dataset):
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

    unit_test = MODELCI_CLIENT.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )
    assert isinstance(unit_test, UnitTest)
    assert unit_test
    assert unit_test.id
    assert unit_test.name
    assert unit_test.slice_id


# TODO: Move unit test to fixture once deletion is implemented
def test_client_unit_test_metric_creation(MODELCI_CLIENT, dataset):
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

    unit_test = MODELCI_CLIENT.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )

    unit_test_metric = MODELCI_CLIENT.create_unit_test_metric(
        unit_test_name=test_name,
        eval_function_name=EVAL_FUNCTION_NAME,
        threshold=EVAL_FUNCTION_THRESHOLD,
        threshold_comparison=ThresholdComparison.GREATER_THAN,
    )
    assert unit_test_metric["unit_test_id"]
    assert unit_test_metric["eval_function_id"]
    assert unit_test_metric["threshold"]
    assert unit_test_metric["threhsold_comparison"]

    metrics = unit_test.get_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) == 1
    assert metrics[0] == unit_test_metric


# TODO: Move unit test to fixture once deletion is implemented
def test_unit_test_metric_creation_from_class(MODELCI_CLIENT, dataset):
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

    unit_test = MODELCI_CLIENT.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )

    unit_test_metric = unit_test.add_metric(
        eval_function_name=EVAL_FUNCTION_NAME,
        threshold=EVAL_FUNCTION_THRESHOLD,
        threshold_comparison=ThresholdComparison.GREATER_THAN,
    )
    assert unit_test_metric["unit_test_id"]
    assert unit_test_metric["eval_function_id"]
    assert unit_test_metric["threshold"]
    assert unit_test_metric["threhsold_comparison"]

    metrics = unit_test.get_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) == 1
    assert metrics[0] == unit_test_metric
