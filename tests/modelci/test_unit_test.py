import pytest

from nucleus.modelci import CreateUnitTestError
from nucleus.modelci.unit_test import UnitTest
from tests.helpers import (
    EVAL_FUNCTION_COMPARISON,
    EVAL_FUNCTION_THRESHOLD,
    get_uuid,
)


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


def test_list_unit_test(CLIENT, dataset, test_slice):
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique

    e = CLIENT.modelci.eval_functions
    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=test_slice.slice_id,
        evaluation_criteria=[e.iou() > 0.5],
    )

    unit_tests = CLIENT.modelci.list_unit_tests()
    assert all(isinstance(unit_test, UnitTest) for unit_test in unit_tests)
    assert unit_test in unit_tests


def test_no_criteria_raises_error(CLIENT, dataset, test_slice):
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    with pytest.raises(CreateUnitTestError):
        CLIENT.modelci.create_unit_test(
            name=test_name,
            slice_id=test_slice.slice_id,
            evaluation_criteria=[],
        )
