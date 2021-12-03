from nucleus.modelci.unit_test import UnitTest
from tests.helpers import (
    EVAL_FUNCTION_COMPARISON,
    EVAL_FUNCTION_THRESHOLD,
    TEST_SLICE_NAME,
    get_uuid,
)
from tests.test_dataset import make_dataset_items


def test_unit_test_metric_creation(CLIENT, unit_test):
    # create some dataset_items for the unit test to reference
    iou = CLIENT.modelci.eval_functions.iou
    unit_test_metric = unit_test.add_criteria(iou() > EVAL_FUNCTION_THRESHOLD)
    assert unit_test_metric.unit_test_id == unit_test.id
    assert unit_test_metric.eval_function_id
    assert unit_test_metric.threshold == EVAL_FUNCTION_THRESHOLD
    assert unit_test_metric.threshold_comparison == EVAL_FUNCTION_COMPARISON

    metrics = unit_test.get_criteria()
    assert isinstance(metrics, list)
    assert unit_test_metric in metrics


def test_list_unit_test(CLIENT, dataset):
    # create some dataset_items for the unit test to reference
    items = make_dataset_items()
    dataset.append(items)
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[items[0].reference_id],
    )

    e = CLIENT.modelci.eval_functions
    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
        evaluation_criteria=[e.iou() > 0.5],
    )

    unit_tests = CLIENT.modelci.list_unit_tests()
    assert all(isinstance(unit_test, UnitTest) for unit_test in unit_tests)
    assert unit_test in unit_tests
